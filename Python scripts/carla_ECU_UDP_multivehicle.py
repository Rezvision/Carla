# carla_3vehicles_ECU.py
# Spawns 3 vehicles in CARLA, sends UDP to central computer, logs data to Parquet/CSV

import carla
import struct
import time
import random
import socket
import sys
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

# --- Configuration ---
UDP_TARGET = "192.168.0.125"

VEHICLE_SIGNAL_PORTS = [
    {"CAN_speed": 5100, "CAN_battery": 5101, "CAN_throttle": 5102, 
     "CAN_brake": 5103, "CAN_steering": 5104, "CAN_gear": 5105},
    {"CAN_speed": 5200, "CAN_battery": 5201, "CAN_throttle": 5202,
     "CAN_brake": 5203, "CAN_steering": 5204, "CAN_gear": 5205},
    {"CAN_speed": 5300, "CAN_battery": 5301, "CAN_throttle": 5302,
     "CAN_brake": 5303, "CAN_steering": 5304, "CAN_gear": 5305},
]

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
    sockets = []
    loc_sockets = []
    for i in range(num):
        sock = {}
        for name, port in VEHICLE_SIGNAL_PORTS[i].items():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock[name] = (s, (UDP_TARGET, port))
        sockets.append(sock)
        loc_sockets.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
    return sockets, loc_sockets


def get_data(vehicle, battery):
    vel = vehicle.get_velocity()
    ctrl = vehicle.get_control()
    loc = vehicle.get_location()
    speed = 3.6 * ((vel.x**2 + vel.y**2)**0.5)
    battery = max(0, battery - ctrl.throttle * 0.1)
    
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


def send_udp(sockets, data, idx):
    signals = [
        ("CAN_speed", data["speed_kmh"]),
        ("CAN_battery", data["battery_level"]),
        ("CAN_throttle", data["throttle"]),
        ("CAN_brake", data["brake"]),
        ("CAN_steering", data["steering"]),
        ("CAN_gear", data["gear"]),
    ]
    for name, val in signals:
        s, addr = sockets[name]
        s.sendto(struct.pack('>f', float(val)), addr)


def send_location(sock, data, idx):
    loc = f"{data['location_x']},{data['location_y']},{data['location_z']}"
    sock.sendto(loc.encode(), (UDP_TARGET, LOCATION_PORTS[idx]))


def cleanup(vehicles, sockets, loc_sockets, loggers):
    for logger in loggers:
        logger.close()
    for sock in sockets:
        for s, _ in sock.values():
            s.close()
    for s in loc_sockets:
        s.close()
    for v in vehicles:
        v.destroy()
    print(f"\nLogs saved to {LOG_DIR}/")


if __name__ == "__main__":
    NUM_VEHICLES = 3
    
    print(f"Starting {NUM_VEHICLES} vehicles, logging to {LOG_DIR}/")
    
    client = connect_to_carla()
    vehicles = spawn_vehicles(client, NUM_VEHICLES)
    sockets, loc_sockets = setup_sockets(NUM_VEHICLES)
    loggers = [DataLogger(i) for i in range(NUM_VEHICLES)]
    batteries = [100.0] * NUM_VEHICLES
    
    try:
        while True:
            for i, vehicle in enumerate(vehicles):
                if not vehicle.is_alive:
                    continue
                data, batteries[i] = get_data(vehicle, batteries[i])
                data['timestamp'] = datetime.now().isoformat()
                
                send_udp(sockets[i], data, i)
                send_location(loc_sockets[i], data, i)
                loggers[i].log(data)
            
            time.sleep(0.05)  # 20 Hz
            
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(vehicles, sockets, loc_sockets, loggers)