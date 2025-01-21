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
FLASK_SERVER_URL = "http://127.0.0.1:5000/update_data"
SNIFF_LOG = "sniffed_data.json"  # Log for sniffed data
REPLAY_BUFFER = []  # Buffer to store packets for replay attack

# Function to log sniffed data
def log_sniffed_data(packet):
    with open(SNIFF_LOG, "a") as file:
        file.write(f"{datetime.now()} - {packet}\n")

# Function to simulate attacks
def simulate_attack(packet, attack_type):
    attack_packet = packet.copy()
    if attack_type == "Masquerade":
        attack_packet["vehicle_model"] = "vehicle.fake.model"  # Fake model
        attack_packet["control"]["brake"] = 1.0  # Fake full braking
    elif attack_type == "Replay":
        if REPLAY_BUFFER:
            attack_packet = random.choice(REPLAY_BUFFER)  # Replay a random packet
    elif attack_type == "DoS":
        for _ in range(10):  # Flood with 10 identical packets
            requests.post(FLASK_SERVER_URL, json=packet)
    return attack_packet

# Function to check for alerts
def check_alerts(packet):
    alerts = []
    if packet["speed_kmh"] > SPEED_LIMIT:
        alerts.append(f"Speed limit exceeded: {packet['speed_kmh']} km/h")
    if packet["control"]["brake"] > HARSH_BRAKE_THRESHOLD:
        alerts.append(f"Harsh braking detected: {packet['control']['brake']} intensity")
    return alerts

# Function to start CARLA simulation
def start_carla():
    try:
        # Connect to the CARLA simulator
        client = carla.Client('localhost', 2000)
        client.set_timeout(3.0)

        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        print(f'Created vehicle: {vehicle.type_id}')
        vehicle.set_autopilot(True)

        start_time = time.time()
        duration = 60  # Run for 60 seconds
        attack_interval = 10  # Every 10 seconds, simulate an attack

        while time.time() - start_time < duration:
            velocity = vehicle.get_velocity()
            speed = (3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5)  # m/s to km/h
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

            # Log normal data for sniffing
            log_sniffed_data(data_packet)
            REPLAY_BUFFER.append(data_packet)

            # Simulate attacks at regular intervals
            if int(time.time() - start_time) % attack_interval == 0:
                attack_type = random.choice(["Masquerade", "Replay", "DoS"])
                print(f"Simulating {attack_type} attack...")
                attack_packet = simulate_attack(data_packet, attack_type)
                data_packet = attack_packet  # Replace with attack packet

            # Send data to Flask server
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

            print(f"CAN Data Packet: {data_packet}")
            print("-" * 40)

            time.sleep(1)

    finally:
        print("Destroying actors...")
        client.apply_batch([carla.command.DestroyActor(x) for x in [vehicle]])
        print("Done.")

# Main entry point
if __name__ == "__main__":
    start_carla()
 