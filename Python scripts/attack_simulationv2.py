import random
import time
import csv
from datetime import datetime
import requests
import carla

# Constants
SPEED_LIMIT = 30.0  # Normal speed limit for the CARLA environment (km/h)
FLASK_SERVER_URL = "http://127.0.0.1:5000/update_data"
DATA_LOG_FILE = "vehicle_data.csv"  # File to store data

# Ensure the CSV file has a header
def initialize_data_file():
    with open(DATA_LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp", "vehicle_model", "speed_kmh",
            "location_x", "location_y", "location_z",
            "throttle", "brake", "steering", "gear",
            "tampered", "tampering_type"
        ])

# Function to simulate tampering attacks
def simulate_tampering(packet):
    """Simulates tampering attacks by modifying speed and/or GPS coordinates."""
    tampered_packet = packet.copy()
    tampering_type = random.choice(["speed", "gps", "both"])

    if tampering_type in ["speed", "both"]:
        # Tamper with speed: Set to an unrealistic value
        tampered_packet["speed_kmh"] = round(random.uniform(31, 50), 2)

    if tampering_type in ["gps", "both"]:
        # Tamper with GPS: Introduce significant unrealistic shifts
        tampered_packet["location"]["x"] += random.uniform(1000, 5000)
        tampered_packet["location"]["y"] += random.uniform(1000, 5000)

    tampered_packet["tampered"] = True  # Mark as tampered
    tampered_packet["tampering_type"] = tampering_type
    return tampered_packet

# Function to log data to a CSV file
def log_data_to_file(packet):
    with open(DATA_LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            packet["timestamp"], packet["vehicle_model"], packet["speed_kmh"],
            packet["location"]["x"], packet["location"]["y"], packet["location"]["z"],
            packet["control"]["throttle"], packet["control"]["brake"], packet["control"]["steering"], packet["control"]["gear"],
            packet["tampered"], packet["tampering_type"]
        ])

# Main function to start the CARLA simulation
def start_carla():
    try:
        # Initialize the data log file
        initialize_data_file()

        # Connect to CARLA simulator
        client = carla.Client('localhost', 2000)
        client.set_timeout(3.0)
        world = client.get_world()

        # Select a Tesla Model 3
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        vehicle.set_autopilot(True)

        # Start simulation
        start_time = time.time()
        duration = 60  # Run for 60 seconds
        print("Starting vehicle simulation...")

        while time.time() - start_time < duration:
            # Collect normal vehicle data
            velocity = vehicle.get_velocity()
            speed = (3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5)
            control = vehicle.get_control()
            location = vehicle.get_location()

            # Generate data packet
            data_packet = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                "tampered": False,  # Default as normal
                "tampering_type": "none",
            }

            # Simulate tampering in ~10% of packets
            if random.random() < 0.1:
                data_packet = simulate_tampering(data_packet)

            # Send data to the Flask server
            try:
                response = requests.post(FLASK_SERVER_URL, json=data_packet)
                if response.status_code == 200:
                    print(f"Data sent: {data_packet}")
                else:
                    print(f"Failed to send data. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error sending data: {e}")

            # Log data to CSV
            log_data_to_file(data_packet)

            # Wait 1 second between packets
            time.sleep(1)

    finally:
        print("Destroying actors...")
        client.apply_batch([carla.command.DestroyActor(x) for x in [vehicle]])
        print("Done.")

if __name__ == "__main__":
    start_carla()
