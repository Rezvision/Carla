# fed_client.py - Federated Learning Edge Client for Raspberry Pi (TFLite)
"""
Edge client using TFLite for inference on Raspberry Pi.
Communicates with central server via MQTT.
"""

import numpy as np
import json
import time
import threading
import queue
import pickle
import os
from collections import deque, OrderedDict
from datetime import datetime
import paho.mqtt.client as mqtt
import struct
import socket

# TFLite runtime for Pi
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        print("WARNING: TFLite not found. Install with: pip install tflite-runtime")
        TFLITE_AVAILABLE = False


class DataCollector:
    """Collects CAN data and device metrics"""
    
    FEATURE_NAMES = [
        'speed_kmh', 'battery_level', 'throttle', 'brake',
        'steering', 'gear', 'location_x', 'location_y'
    ]
    
    def __init__(self, can_interface='can0', buffer_size=1000):
        self.can_interface = can_interface
        self.buffer = deque(maxlen=buffer_size)
        self.current_state = {name: 0.0 for name in self.FEATURE_NAMES}
        self.running = False
        self.message_count = 0
    
    def start(self):
        self.running = True
        self.can_thread = threading.Thread(target=self._can_listener, daemon=True)
        self.can_thread.start()
        print(f"Data collector started on {self.can_interface}")
    
    def stop(self):
        self.running = False
    
    def _can_listener(self):
        try:
            import can
            bus = can.interface.Bus(channel=self.can_interface, interface='socketcan')
            
            while self.running:
                msg = bus.recv(timeout=1.0)
                if msg:
                    self._process_can_message(msg)
        except Exception as e:
            print(f"CAN error: {e}, using simulated data")
            self._simulate_data()
    
    def _process_can_message(self, msg):
        """Decode CAN message"""
        can_id = msg.arbitration_id
        
        try:
            value = struct.unpack('<f', msg.data[:4])[0]
            
            if can_id == 0x123:
                self.current_state['speed_kmh'] = value
            elif can_id == 0x124:
                self.current_state['battery_level'] = value
            elif can_id == 0x125:
                self.current_state['throttle'] = value
            elif can_id == 0x126:
                self.current_state['brake'] = value
            elif can_id == 0x127:
                self.current_state['steering'] = value
            elif can_id == 0x128:
                self.current_state['gear'] = value
            
            self._add_to_buffer()
            self.message_count += 1
        except:
            pass
    
    def _simulate_data(self):
        """Fallback simulated data"""
        import random
        while self.running:
            self.current_state['speed_kmh'] = max(0, self.current_state.get('speed_kmh', 50) + random.gauss(0, 2))
            self.current_state['battery_level'] = max(0, min(100, self.current_state.get('battery_level', 80) - 0.01))
            self.current_state['throttle'] = max(0, min(1, 0.3 + random.gauss(0, 0.1)))
            self.current_state['brake'] = max(0, min(1, random.gauss(0, 0.05)))
            self.current_state['steering'] = random.gauss(0, 0.1)
            self.current_state['gear'] = random.randint(0, 5)
            self._add_to_buffer()
            time.sleep(0.1)
    
    def _add_to_buffer(self):
        features = [self.current_state.get(name, 0.0) for name in self.FEATURE_NAMES]
        self.buffer.append(np.array(features, dtype=np.float32))
    
    def get_sequence(self, seq_length=20):
        """Get recent data as sequence"""
        if len(self.buffer) < seq_length:
            return None
        recent = list(self.buffer)[-seq_length:]
        return np.array(recent, dtype=np.float32)
    
    def get_training_data(self, seq_length=20):
        """Get all buffered data as sequences"""
        if len(self.buffer) < seq_length:
            return None
        
        data = np.array(list(self.buffer), dtype=np.float32)
        num_seq = len(data) - seq_length + 1
        sequences = np.array([data[i:i+seq_length] for i in range(0, num_seq, 5)])
        return sequences


class TFLiteModel:
    """TFLite model wrapper for inference"""
    
    def __init__(self, model_path=None):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.threshold = 0.5
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def load(self, model_path):
        """Load TFLite model"""
        if not TFLITE_AVAILABLE:
            print("TFLite not available")
            return False
        
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Loaded model: {model_path}")
        return True
    
    def predict(self, data):
        """Run inference"""
        if self.interpreter is None:
            return None, 0.0
        
        # Ensure correct shape and type
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        data = data.astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Reconstruction error as anomaly score
        error = np.mean((data - output) ** 2)
        is_anomaly = error > self.threshold
        
        return is_anomaly, float(error)


class FederatedClient:
    """Federated learning client for edge device"""
    
    def __init__(self, client_id, broker, port=1883, can_interface='can0'):
        self.client_id = client_id
        self.broker = broker
        self.port = port
        
        # Components
        self.data_collector = DataCollector(can_interface)
        self.model = TFLiteModel()
        
        # MQTT
        self.mqtt_client = None
        self.connected = False
        
        # State
        self.current_round = 0
        self.model_queue = queue.Queue()
        self.running = False
        
        # Training data stats (for weighted aggregation)
        self.num_samples = 0
        self.local_loss = 0.0
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.mqtt_client = mqtt.Client(
                client_id=f"fed_client_{self.client_id}",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
        except:
            self.mqtt_client = mqtt.Client(client_id=f"fed_client_{self.client_id}")
        
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.on_disconnect = self._on_disconnect
        
        print(f"Connecting to MQTT broker {self.broker}:{self.port}...")
        self.mqtt_client.connect(self.broker, self.port)
        self.mqtt_client.loop_start()
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if hasattr(rc, 'value'):
            rc = rc.value
        
        if rc == 0:
            print(f"Connected to MQTT broker")
            self.connected = True
            
            client.subscribe("federated/model/global")
            client.subscribe("federated/aggregation/trigger")
            client.subscribe(f"federated/clients/{self.client_id}/commands")
            
            self._publish_status("online")
        else:
            print(f"MQTT connection failed: {rc}")
    
    def _on_message(self, client, userdata, message):
        topic = message.topic
        
        try:
            if topic == "federated/model/global":
                model_data = pickle.loads(message.payload)
                self.model_queue.put(model_data)
                print(f"Received global model (round {model_data.get('round', '?')})")
                
            elif topic == "federated/aggregation/trigger":
                trigger = json.loads(message.payload)
                print(f"Training trigger received")
                self._do_local_training()
                
        except Exception as e:
            print(f"Message error: {e}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        self.connected = False
        print("Disconnected from MQTT")
    
    def _publish_status(self, status):
        if self.mqtt_client and self.connected:
            payload = json.dumps({
                'client_id': self.client_id,
                'status': status,
                'timestamp': time.time(),
                'round': self.current_round,
                'num_samples': self.num_samples,
            })
            self.mqtt_client.publish("federated/clients/status", payload)
    
    def _do_local_training(self):
        """Perform local training and send update"""
        self._publish_status("training")
        
        # Get training data
        data = self.data_collector.get_training_data(seq_length=20)
        if data is None or len(data) < 10:
            print("Insufficient data for training")
            return
        
        self.num_samples = len(data)
        
        # Compute reconstruction error on local data (as proxy for loss)
        if self.model.interpreter:
            errors = []
            for seq in data[:100]:  # Sample for speed
                _, error = self.model.predict(seq)
                errors.append(error)
            self.local_loss = np.mean(errors)
        else:
            self.local_loss = 0.0
        
        # Send metrics to server (model weights sent separately if using PyTorch on server)
        metrics = {
            'client_id': self.client_id,
            'round': self.current_round,
            'num_samples': self.num_samples,
            'loss': self.local_loss,
            'timestamp': time.time(),
        }
        
        self.mqtt_client.publish("federated/model/updates", pickle.dumps(metrics))
        print(f"Sent update: samples={self.num_samples}, loss={self.local_loss:.4f}")
        
        self._publish_status("idle")
    
    def _apply_global_model(self, model_data):
        """Apply received model"""
        self.current_round = model_data.get('round', self.current_round + 1)
        
        # If server sends TFLite model bytes, save and load
        if 'tflite_model' in model_data:
            model_path = f"/tmp/model_round{self.current_round}.tflite"
            with open(model_path, 'wb') as f:
                f.write(model_data['tflite_model'])
            self.model.load(model_path)
        
        print(f"Applied global model (round {self.current_round})")
    
    def detect_anomaly(self):
        """Run anomaly detection on recent data"""
        seq = self.data_collector.get_sequence(seq_length=20)
        if seq is None:
            return False, 0.0
        
        return self.model.predict(seq)
    
    def run(self):
        """Main loop"""
        print(f"Starting federated client: {self.client_id}")
        
        self.data_collector.start()
        self.connect()
        
        time.sleep(2)
        self.running = True
        
        try:
            while self.running:
                # Check for new global model
                try:
                    model_data = self.model_queue.get_nowait()
                    self._apply_global_model(model_data)
                except queue.Empty:
                    pass
                
                # Anomaly detection
                if self.model.interpreter:
                    is_anomaly, score = self.detect_anomaly()
                    if is_anomaly:
                        print(f"⚠️ ANOMALY: score={score:.4f}")
                        alert = {
                            'client_id': self.client_id,
                            'type': 'anomaly',
                            'score': score,
                            'timestamp': time.time(),
                        }
                        self.mqtt_client.publish("federated/alerts/attack", json.dumps(alert))
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.data_collector.stop()
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Edge Client')
    parser.add_argument('--client-id', default='edge_1', help='Client ID')
    parser.add_argument('--broker', required=True, help='MQTT broker IP')
    parser.add_argument('--port', type=int, default=1883, help='MQTT port')
    parser.add_argument('--can-interface', default='can0', help='CAN interface')
    
    args = parser.parse_args()
    
    client = FederatedClient(
        client_id=args.client_id,
        broker=args.broker,
        port=args.port,
        can_interface=args.can_interface
    )
    client.run()


if __name__ == "__main__":
    main()
