# carla_direct_ECU.py - Sends raw binary floats via UDP
import carla
import struct
import time
import random
import socket
import sys

# UDP ports: one per signal
SIGNAL_PORT_MAP = {
    "CAN_speed": 5100,
    "CAN_battery": 5101,
    "CAN_throttle": 5102,
    "CAN_brake": 5103,
    "CAN_steering": 5104,
    "CAN_gear": 5105,
}

UDP_TARGET = "127.0.0.1"

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
    speed = 3.6 * ((velocity.x ** 2 + velocity.y ** 2) ** 0.5)
    control = vehicle.get_control()
    location = vehicle.get_location()
    battery_level = update_battery(vehicle, battery_level)
    return {
        "speed_kmh": speed,
        "battery_level": battery_level,
        "throttle": control.throttle,
        "brake": control.brake,
        "steering": control.steer,
        "gear": control.gear,
        "updated_battery_level": battery_level,
        "location": (location.x, location.y, location.z)
    }

def setup_udp_sockets():
    """Create a UDP socket per ECU port."""
    sockets = {}
    for signal_name, port in SIGNAL_PORT_MAP.items():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sockets[signal_name] = (s, (UDP_TARGET, port))
    return sockets

def send_to_ecus_via_udp(sockets, vehicle, battery_level):
    """
    Send raw big-endian binary floats via UDP to each ECU.
    Format: 4 bytes, big-endian IEEE 754 float (network byte order)
    """
    data = get_vehicle_data(vehicle, battery_level)

    # Map signal names to values
    signal_values = {
        "CAN_speed": data["speed_kmh"],
        "CAN_battery": data["battery_level"],
        "CAN_throttle": data["throttle"],
        "CAN_brake": data["brake"],
        "CAN_steering": data["steering"],
        "CAN_gear": float(data["gear"]),  # Convert int to float for consistency
    }

    for signal_name, value in signal_values.items():
        try:
            s, addr = sockets[signal_name]
            # Pack as big-endian float ('>f' = network byte order)
            payload = struct.pack('>f', float(value))
            s.sendto(payload, addr)
            
            # Debug: show hex bytes being sent
            hex_bytes = ' '.join(f'{b:02X}' for b in payload)
            print(f"[UDP] {signal_name} -> port {addr[1]}: {value:.2f} [{hex_bytes}]")
        except Exception as e:
            print(f"Failed UDP send for {signal_name}: {e}", file=sys.stderr)

    return data["updated_battery_level"]

def setup_location_udp_socket():
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_location_data(udp_socket, vehicle, target_ip="192.168.0.111", target_port=9876):
    location = vehicle.get_location()
    location_data = f"{location.x},{location.y},{location.z}"
    udp_socket.sendto(location_data.encode(), (target_ip, target_port))
    # print(f"Sent Location: {location_data}")

def clean_up(vehicle, sockets, udp_socket):
    print("Shutting down...")
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
        print("Starting CARLA -> ECU UDP sender (binary format)...")
        print("Sending big-endian floats to BUSMASTER ECU nodes")
        print("-" * 50)
        while True:
            battery_level = send_to_ecus_via_udp(sockets, vehicle, battery_level)
            send_location_data(udp_socket, vehicle)
            time.sleep(0.05)  # 20 Hz
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        clean_up(vehicle, sockets, udp_socket)