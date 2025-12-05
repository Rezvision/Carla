# carla_direct_ECU.py  (edited)
import carla
import struct
from datetime import datetime
import time
import random
import socket  # For UDP communication
import sys

# --- UDP ports: one per ECU ---
ECU_PORT_MAP = {
    0x123: 5100,  # speed
    0x124: 5101,  # battery
    0x125: 5102,  # throttle
    0x126: 5103,  # brake
    0x127: 5104,  # steering
    0x128: 5105,  # gear (int)
}

UDP_TARGET = ("127.0.0.1",)  # each send will use the port from ECU_PORT_MAP

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
    for can_id, port in ECU_PORT_MAP.items():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sockets[can_id] = (s, ('127.0.0.1', port))
    return sockets

def send_to_ecus_via_udp(sockets, vehicle, battery_level):
    """
    Pack values exactly as before (big-endian floats/ints),
    then send the raw bytes via UDP to each ECU listener.
    """
    data = get_vehicle_data(vehicle, battery_level)

    # Build frames same as original
    frames_data = [
        (0x123, struct.pack('>f', data["speed_kmh"])),
        (0x124, struct.pack('>f', data["battery_level"])),
        (0x125, struct.pack('>f', data["control"]["throttle"])),
        (0x126, struct.pack('>f', data["control"]["brake"])),
        (0x127, struct.pack('>f', data["control"]["steering"])),
        (0x128, struct.pack('>i', data["control"]["gear"])),
    ]

    # Send each to its ECU socket
    for can_id, packed_data in frames_data:
        try:
            s, addr = sockets[can_id]
            # Optionally reserve MSB of last byte = 0 as before
            modified_packed = bytearray(packed_data)
            modified_packed[-1] = modified_packed[-1] & 0x7F
            s.sendto(bytes(modified_packed), addr)
            # For diagnostics print what we sent
            val = (struct.unpack('>f', packed_data)[0]
                   if can_id != 0x128 else struct.unpack('>i', packed_data)[0])
            print(f"[UDP->ECU] ID={hex(can_id)} port={addr[1]} value={val}")
        except Exception as e:
            print(f"Failed UDP send to ECU for ID {hex(can_id)}: {e}", file=sys.stderr)

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
