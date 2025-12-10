# carla_direct_ECU.py  (edited)
import carla
import struct
import json
from datetime import datetime
import time
import random
import socket  # For UDP communication
import sys

# --- UDP ports: one per signal name. We send messages that contain only the
# signal name and its value (no CAN ID in the UDP payload). Example signal
# names: 'CAN_speed', 'CAN_battery', etc.
SIGNAL_PORT_MAP = {
    "CAN_speed": 5100,
    "CAN_battery": 5101,
    "CAN_throttle": 5102,
    "CAN_brake": 5103,
    "CAN_steering": 5104,
    "CAN_gear": 5105,
}

UDP_TARGET = "192.168.0.125"  # each send will use the port from ECU_PORT_MAP

def connect_to_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)
    return client

def spawn_vehicle(client):
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)
    transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, transform)
    vehicle.set_autopilot(True)
    print(f"Spawned vehicle: {vehicle.type_id}")
    return vehicle

def update_battery(vehicle, battery_level):
    throttle = vehicle.get_control().throttle
    battery_level -= throttle * 0.1
    return max(0, battery_level)

def get_vehicle_data(vehicle, battery_level):
    velocity = vehicle.get_velocity()
    speed = 3.6 * ((velocity.x ** 2 + velocity.y ** 2 ) ** 0.5)
    control = vehicle.get_control()
    location = vehicle.get_location()
    battery_level = update_battery(vehicle, battery_level)
    return {
        "speed_kmh": round(speed, 2),
        "battery_level": round(battery_level, 2),
        "control": {
            "throttle": round(control.throttle, 2),
            "brake": round(control.brake, 2),
            "steering": round(control.steer, 2),
            "gear": control.gear,
        },
        "updated_battery_level": battery_level,
        "location": (location.x, location.y, location.z)
    }

def setup_udp_sockets():
    """Create a UDP socket per ECU port (for faster sends)."""
    sockets = {}
    for signal_name, port in SIGNAL_PORT_MAP.items():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sockets[signal_name] = (s, (UDP_TARGET, port))
    return sockets

def send_to_ecus_via_udp(sockets, vehicle, battery_level):
    """
    Pack values exactly as before (big-endian floats/ints),
    then send the raw bytes via UDP to each ECU listener.
    """
    data = get_vehicle_data(vehicle, battery_level)

    # Build list of (signal_name, value). We will send a small JSON object
    # containing the signal name and the value. No CAN ID is included.
    frames_data = [
        ("CAN_speed", data["speed_kmh"]),
        ("CAN_battery", data["battery_level"]),
        ("CAN_throttle", data["control"]["throttle"]),
        ("CAN_brake", data["control"]["brake"]),
        ("CAN_steering", data["control"]["steering"]),
        ("CAN_gear", data["control"]["gear"]),
    ]

    # Send each signal as a JSON payload: {"signal":"CAN_speed","value":123.45}
    for signal_name, value in frames_data:
        try:
            s, addr = sockets[signal_name]
            payload = json.dumps({"signal": signal_name, "value": value}).encode()
            s.sendto(payload, addr)
            print(f"[UDP->ECU] signal={signal_name} port={addr[1]} value={value}")
        except Exception as e:
            print(f"Failed UDP send for signal {signal_name}: {e}", file=sys.stderr)

    return data["updated_battery_level"]

def setup_location_udp_socket():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return udp_socket

def send_location_data(udp_socket, vehicle, target_ip="192.168.0.111", target_port=9876):
    location = vehicle.get_location()
    location_data = f"{location.x},{location.y},{location.z}"
    udp_socket.sendto(location_data.encode(), (target_ip, target_port))
    print(f"Sent Location Data: {location_data} to {target_ip}:{target_port}")

def receive_feature_can_dummy():  # kept as placeholder if you still have gs-usb input logic
    # Not used in UDP-driven flow, keep for later if you want to accept commands from CAN hardware.
    return

def clean_up(vehicle, sockets, udp_socket):
    print("Shutting down sockets and destroying actors...")
    for s, _ in sockets.values():
        try:
            s.close()
        except:
            pass
    try:
        udp_socket.close()
    except:
        pass
    vehicle.destroy()
    print("Cleaned up.")

if __name__ == "__main__":
    client = connect_to_carla()
    vehicle = spawn_vehicle(client)

    sockets = setup_udp_sockets()
    udp_socket = setup_location_udp_socket()
    battery_level = 100.0

    try:
        print("Starting CARLA -> ECU UDP sender...")
        while True:
            battery_level = send_to_ecus_via_udp(sockets, vehicle, battery_level)
            send_location_data(udp_socket, vehicle)
            # You can reduce or increase sleep. This drives how often ECUs receive updates.
            time.sleep(0.05)  # 20 Hz update from CARLA
    except KeyboardInterrupt:
        print("Keyboard interrupt, stopping...")
    finally:
        clean_up(vehicle, sockets, udp_socket)
