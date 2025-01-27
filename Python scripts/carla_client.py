import glob
import os
import sys
import random
import time
import carla
from datetime import datetime

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

def get_vehicle_data(vehicle):
    velocity = vehicle.get_velocity()
    speed = (3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5)
    control = vehicle.get_control()
    location = vehicle.get_location()

    data_packet = {
        "timestamp": f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        "vehicle_model": vehicle.type_id,
        "speed_kmh": round(speed, 2),
        "location": {
            "x": round(location.x, 2),
            "y": round(location.y, 2),
            "z": round(location.z, 2),
        },
        "control": {
            "throttle": round(control.throttle, 2),
            "brake": round(control.brake, 2),
            "steering": round(control.steer, 2),
            "gear": control.gear,
        },
    }
    return data_packet
def battery_level(vehicle):
    battery_level = vehicle.get_battery_level()
    print(f'Battery level: {battery_level}')
    return battery_level
def clean_up(vehicle):
    print("Destroying actors...")
    vehicle.destroy()
    print("Actors destroyed.")

if __name__ == "__main__":
    from vTCU_python import TCU

    FLASK_SERVER_URL = "http://127.0.0.1:5000/update_data"

    tcu = TCU(FLASK_SERVER_URL)
    client = connect_to_carla()
    vehicle = spawn_vehicle(client)

    try:
        print("Starting CARLA and TCU simulation...")
        start_time = time.time()
        duration = 60

        while time.time() - start_time < duration:
            packet = get_vehicle_data(vehicle)
            tcu.store_packet(packet)
            tcu.send_to_cloud(packet)

            # Simulate poor connection handling every 10 seconds
            if int(time.time() - start_time) % 10 == 0:
                tcu.simulate_poor_connection()

            time.sleep(1)
    finally:
        clean_up(vehicle)
