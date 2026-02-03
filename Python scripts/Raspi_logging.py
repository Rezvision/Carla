import time
import subprocess

# ----------------------------
# CPU usage helper
# ----------------------------
def read_cpu_times():
    with open("/proc/stat", "r") as f:
        fields = f.readline().strip().split()[1:]
        fields = list(map(int, fields))
    idle = fields[3] + fields[4]
    total = sum(fields)
    return idle, total


def cpu_usage(prev_idle, prev_total, idle, total):
    idle_delta = idle - prev_idle
    total_delta = total - prev_total
    return 100.0 * (1.0 - idle_delta / total_delta)


# ----------------------------
# Temperature
# ----------------------------
def cpu_temperature():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        return int(f.read()) / 1000.0


# ----------------------------
# Memory
# ----------------------------
def memory_usage():
    meminfo = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":")
            meminfo[key] = int(value.strip().split()[0])
    total = meminfo["MemTotal"] / 1024
    free = meminfo["MemAvailable"] / 1024
    used = total - free
    return used, total


# ----------------------------
# Load average
# ----------------------------
def load_average():
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


# ----------------------------
# Throttling status (Pi)
# ----------------------------
def throttling_status():
    """0x0 → No issues

0x2 → CPU frequency capped now

0x20000 → Throttling occurred in the past"""

    try:
        output = subprocess.check_output(["vcgencmd", "get_throttled"])
        return output.decode().strip()
    except Exception:
        return "vcgencmd unavailable"


# ----------------------------
# Main loop
# ----------------------------
if __name__ == "__main__":
    prev_idle, prev_total = read_cpu_times()

    while True:
        time.sleep(1)

        idle, total = read_cpu_times()
        cpu = cpu_usage(prev_idle, prev_total, idle, total)
        prev_idle, prev_total = idle, total

        temp = cpu_temperature()
        mem_used, mem_total = memory_usage()
        load = load_average()
        throttled = throttling_status()

        print(
            f"CPU: {cpu:5.1f}% | "
            f"Temp: {temp:4.1f}°C | "
            f"Mem: {mem_used:.0f}/{mem_total:.0f} MB | "
            f"Load: {load} | "
            f"{throttled}"
        )
