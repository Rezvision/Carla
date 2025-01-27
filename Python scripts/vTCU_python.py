import time
import json
import requests

class TCU:
    def __init__(self, flask_server_url):
        self.memory_module = []
        self.last_sent_packet = None
        self.flask_server_url = flask_server_url
        self.data_communication = {
            "gnss": None,
            "vehiclecan_in": None,
            "vehiclecan_out": None,
            "cloud_in": None,
            "cloud_out": None,
        }

    def store_packet(self, packet):
        self.memory_module.append(packet)

    def send_to_cloud(self, packet):
        if packet != self.last_sent_packet:  # Only send if data has changed! however this needs to pay attention diffrerent signals because obviouslt each time stamp will result in a differtn data packet
            try:
                response = requests.post(self.flask_server_url, json=packet)
                if response.status_code == 200:
                    print(f"Data sent to cloud: {packet}")
                    self.last_sent_packet = packet
                else:
                    print(f"Failed to send data to cloud. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error sending data to cloud: {e}")

    def simulate_poor_connection(self):
        # Simulate poor connection by sending stored data after delay
        print("Simulating poor connection. Sending stored data...")
        for packet in self.memory_module:
            self.send_to_cloud(packet)
        self.memory_module = []  # Clear memory after sending