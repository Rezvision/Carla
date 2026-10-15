#!/usr/bin/env python3
"""
carla_fdx_sender.py  --  CARLA -> CANoe (FDX) telemetry sender + logger.

Feature-parity port of carla_ECU_UDP_multivehicle.py. Everything is identical
(3 vehicles, battery model, 20 Hz loop, Parquet/CSV logging, location -> Pi)
EXCEPT the CAN-signal transport: the six per-vehicle signals that used to be
sent as per-signal UDP packets to the central computer are now injected into
CANoe via the FDX protocol, keyed by groupID.

FDX groupID map (matches FDXDescription_AllVehicles.xml):
    V1: speed=1  battery=2  throttle=3  brake=4  steering=5  gear=6
    V2: speed=7  battery=8  throttle=9  brake=10 steering=11 gear=12
    V3: speed=13 battery=14 throttle=15 brake=16 steering=17 gear=18
CANoe resolves groupID -> namespace (FDX / FDX_2 / FDX_3) internally, so on the
wire only the groupID matters. All values are 8-byte little-endian doubles
(gear included), exactly as in the original FDX demo sender.

Location is still sent as UDP to the edge Pis: it isn't a CAN signal and has no
FDX group, so the FDX substitution doesn't apply to it.
"""

import carla
import struct
import time
import random
import socket
import os
from datetime import datetime

# Try parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET = True
except ImportError:
    PARQUET = False
    import csv

# --- FDX / CANoe configuration ---
CANOE_IP   = "192.168.0.125"   # host running CANoe. Default assumes CANoe is on the
CANOE_PORT = 5100         # same workstation as CARLA (as in the demo sender).
                           # If CANoe runs on the central computer instead, set
                           # this to "192.168.0.125". Port must match CANoe's
                           # FDX options.

SIGNATURE = 0x584446656F4E4143   # "CANoeFDX"
MAJOR, MINOR = 1, 0
CMD_DATA_EXCHANGE = 0x0005

# (vehicle, signal) -> FDX groupID
GROUPS = {
    (1, "speed"): 1,   (1, "battery"): 2,   (1, "throttle"): 3,
    (1, "brake"): 4,   (1, "steering"): 5,  (1, "gear"): 6,
    (2, "speed"): 7,   (2, "battery"): 8,   (2, "throttle"): 9,
    (2, "brake"): 10,  (2, "steering"): 11, (2, "gear"): 12,
    (3, "speed"): 13,  (3, "battery"): 14,  (3, "throttle"): 15,
    (3, "brake"): 16,  (3, "steering"): 17, (3, "gear"): 18,
}

# --- Location (edge Pi) configuration -- unchanged, still UDP ---
PI_IPS = ["192.168.0.111", "192.168.0.211", "192.168.0.99"]  # edge_1, edge_2, edge_3
LOCATION_PORTS = [9876, 9877, 9878]  # Different port per vehicle

LOG_DIR = "./vehicle_logs"
LOG_BUFFER_SIZE = 500


class DataLogger:
    """Simple logger - saves raw data to Parquet or CSV"""

    COLUMNS = ['timestamp', 'speed_kmh', 'battery_level', 'throttle', 'brake',
               'steering', 'gear', 'location_x', 'location_y', 'location_z']

    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.buffer = []

        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "parquet" if PARQUET else "csv"
        self.filename = os.path.join(LOG_DIR, f"vehicle_{vehicle_id+1}_{ts}.{ext}")

        if not PARQUET:
            self.csv_file = open(self.filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.COLUMNS)
            self.csv_writer.writeheader()

    def log(self, data):
        self.buffer.append(data)
        if len(self.buffer) >= LOG_BUFFER_SIZE:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return

        if PARQUET:
            table = pa.Table.from_pylist(self.buffer)
            if os.path.exists(self.filename):
                existing = pq.read_table(self.filename)
                table = table.cast(existing.schema)
                table = pa.concat_tables([existing, table])
            pq.write_table(table, self.filename, compression='snappy')
        else:
            for row in self.buffer:
                self.csv_writer.writerow(row)
            self.csv_file.flush()

        self.buffer.clear()

    def close(self):
        self._flush()
        if not PARQUET and hasattr(self, 'csv_file'):
            self.csv_file.close()


class FDXSender:
    def __init__(self, ip=CANOE_IP, port=CANOE_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (ip, port)
        self.seq = 0
        self.sent = 0            # datagrams the OS accepted
        self.bytes_sent = 0
        self.errors = 0
        self._seen = set()       # groupIDs already confirmed once
        self._warned = False     # print the "no listener" warning only once

    def _next_seq(self):
        self.seq = (self.seq + 1) & 0xFFFF
        return self.seq

    def _send(self, group_id, value):
        data = struct.pack("<d", float(value))
        header = struct.pack("<QBBHHH", SIGNATURE, MAJOR, MINOR, 1, self._next_seq(), 0)
        cmd = struct.pack("<HHHH", 8 + len(data), CMD_DATA_EXCHANGE, group_id, len(data))
        pkt = header + cmd + data
        try:
            n = self.sock.sendto(pkt, self.addr)
            self.sent += 1
            self.bytes_sent += n
            if n != len(pkt):
                print(f"[FDX] WARN groupID {group_id}: only {n}/{len(pkt)} bytes sent")
            elif group_id not in self._seen:           # first datagram for this group
                self._seen.add(group_id)
                print(f"[FDX] groupID {group_id:>2} first send OK "
                      f"({n} bytes -> {self.addr[0]}:{self.addr[1]}, value={value})")
        except ConnectionResetError:
            self.errors += 1
            if not self._warned:
                self._warned = True
                print(f"[FDX] !! nothing listening at {self.addr[0]}:{self.addr[1]} -- "
                      f"is CANoe measuring, and is its FDX port set to {self.addr[1]}?")
        except OSError as e:
            self.errors += 1
            print(f"[FDX] send error, groupID {group_id}: {e}")

    def send_vehicle(self, vehicle, data):
        """Send all six signals for one vehicle. `vehicle` is 1-indexed to
        match the groupID map."""
        signals = [
            ("speed",    data["speed_kmh"]),
            ("battery",  data["battery_level"]),
            ("throttle", data["throttle"]),
            ("brake",    data["brake"]),
            ("steering", data["steering"]),
            ("gear",     data["gear"]),
        ]
        for ecu, val in signals:
            self._send(GROUPS[(vehicle, ecu)], val)

    def close(self):
        self.sock.close()


def connect_to_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    return client


def spawn_vehicles(client, num=3):
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()

    # Get all vehicle blueprints and pick randomly
    all_vehicles = bp_lib.filter('vehicle.*')

    vehicles = []
    for i in range(num):
        bp = random.choice(all_vehicles)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))

        vehicle = world.spawn_actor(bp, spawns[i % len(spawns)])
        vehicle.set_autopilot(True)
        vehicles.append(vehicle)
        print(f"[Vehicle {i+1}] Spawned {bp.id}")

    return vehicles


def setup_sockets(num):
    """One shared FDX sender for all vehicles + one location UDP socket each."""
    fdx = FDXSender()
    loc_sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(num)]
    return fdx, loc_sockets


def get_data(vehicle, battery):
    vel = vehicle.get_velocity()
    ctrl = vehicle.get_control()
    loc = vehicle.get_location()
    speed = 3.6 * ((vel.x**2 + vel.y**2)**0.5)
    battery = max(0, battery - ctrl.throttle * 0.000298)

    return {
        'speed_kmh': round(speed, 2),
        'battery_level': float(round(battery, 2)),
        'throttle': round(ctrl.throttle, 2),
        'brake': round(ctrl.brake, 2),
        'steering': round(ctrl.steer, 2),
        'gear': ctrl.gear,
        'location_x': round(loc.x, 2),
        'location_y': round(loc.y, 2),
        'location_z': round(loc.z, 2),
    }, battery


def send_location(sock, data, idx):
    loc = f"{data['location_x']},{data['location_y']},{data['location_z']}"
    sock.sendto(loc.encode(), (PI_IPS[idx], LOCATION_PORTS[idx]))


def cleanup(vehicles, fdx, loc_sockets, loggers):
    for logger in loggers:
        logger.close()
    fdx.close()
    for s in loc_sockets:
        s.close()
    for v in vehicles:
        v.destroy()
    print(f"\nLogs saved to {LOG_DIR}/")


if __name__ == "__main__":
    NUM_VEHICLES = 3

    print(f"Starting {NUM_VEHICLES} vehicles -> CANoe FDX at {CANOE_IP}:{CANOE_PORT}")
    print(f"Logging to {LOG_DIR}/")

    client = connect_to_carla()
    vehicles = spawn_vehicles(client, NUM_VEHICLES)
    fdx, loc_sockets = setup_sockets(NUM_VEHICLES)
    loggers = [DataLogger(i) for i in range(NUM_VEHICLES)]
    batteries = [100.0] * NUM_VEHICLES

    ticks = 0
    try:
        while True:
            for i, vehicle in enumerate(vehicles):
                if not vehicle.is_alive:
                    continue
                data, batteries[i] = get_data(vehicle, batteries[i])
                data['timestamp'] = datetime.now().isoformat()

                fdx.send_vehicle(i + 1, data)      # FDX groupIDs are 1-indexed
                send_location(loc_sockets[i], data, i)
                loggers[i].log(data)

            ticks += 1
            if ticks % 200 == 0:      # ~every 10 s at 20 Hz
                print(f"[{datetime.now():%H:%M:%S}] tick {ticks} | "
                      f"FDX sent={fdx.sent} bytes={fdx.bytes_sent} errors={fdx.errors}")

            time.sleep(0.05)  # 20 Hz

    except KeyboardInterrupt:
        pass
    finally:
        cleanup(vehicles, fdx, loc_sockets, loggers)
