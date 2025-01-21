# carla_simulation.py
import glob
import os
import sys
import random
import time
from datetime import datetime
import requests  # To send HTTP requests to Flask server
import carla

# Constants
SPEED_LIMIT = 100.0  # km/h
HARSH_BRAKE_THRESHOLD = 0.7  # Brake intensity
DATA_PACKET = {}

# Flask server URL
FLASK_SERVER_URL = "http://127.0.0.1:5000/update_data"

# Function to check for custom alerts
def check_alerts(packet):
    alerts = []
    if packet["speed_kmh"] > SPEED_LIMIT:
        alerts.append(f"Speed limit exceeded: {packet['speed_kmh']} km/h")
    if packet["control"]["brake"] > HARSH_BRAKE_THRESHOLD:
        alerts.append(f"Harsh braking detected: {packet['control']['brake']} intensity")
    return alerts

# Function to simulate vehicle data generation (CARLA simulation)
def start_carla():
    try:
        # Connect to the CARLA simulator
        client = carla.Client('localhost', 2000)
        client.set_timeout(3.0)

        # Get the world
        world = client.get_world()

        # Get the blueprint library and select a Tesla Model 3
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        # Randomize the vehicle color if applicable
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Choose a random spawn point
        transform = random.choice(world.get_map().get_spawn_points())

        # Spawn the vehicle
        vehicle = world.spawn_actor(bp, transform)
        print(f'Created vehicle: {vehicle.type_id}')

        # Enable autopilot for the vehicle
        vehicle.set_autopilot(True)

        # Start the timer
        start_time = time.time()
        duration = 60  # Run for 60 seconds

        print("Starting vehicle simulation...")

        while time.time() - start_time < duration:
            # Get vehicle data
            velocity = vehicle.get_velocity()
            speed = (3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5)  # Convert m/s to km/h
            control = vehicle.get_control()
            location = vehicle.get_location()

            # Create a CAN-like data packet
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

            # Send data packet to Flask server
            try:
                response = requests.post(FLASK_SERVER_URL, json=data_packet)
                if response.status_code == 200:
                    print(f"Data sent to Flask: {data_packet}")
                else:
                    print(f"Failed to send data to Flask. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error sending data to Flask: {e}")

            # Check for alerts
            alerts = check_alerts(data_packet)
            if alerts:
                for alert in alerts:
                    print(f"ALERT: {alert}")

            # Display the CAN-like data packet
            print(f"CAN Data Packet: {data_packet}")
            print("-" * 40)

            # Wait for 1 second
            time.sleep(1)

    finally:
        print("Destroying actors...")
        client.apply_batch([carla.command.DestroyActor(x) for x in [vehicle]])
        print("Done.")

# Main entry point to start CARLA simulation
if __name__ == "__main__":
    start_carla()
