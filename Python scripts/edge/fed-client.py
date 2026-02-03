# fed_client.py - Federated Learning Edge Client for Raspberry Pi
"""
Edge client for federated learning IDS.
Runs on Raspberry Pi devices, performs local training and communicates with server.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
import threading
import queue
import pickle
import io
import base64
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict, deque
from dataclasses import dataclass, field
import paho.mqtt.client as mqtt
import struct
import socket

# Local imports
try:
    from edge_model import EdgeAutoencoder, create_edge_model
    from privacy import (
        InputPerturbation, OutputPerturbation, DPSGDOptimizer,
        PrivacyMechanism, create_privacy_mechanism
    )
    from config import (
        edge_model_config, federated_config, privacy_config,
        device_config, can_config
    )
except ImportError:
    print("Warning: Some modules not found. Using default configurations.")


@dataclass
class ClientConfig:
    """Edge client configuration"""
    client_id: str = "edge_1"
    mqtt_broker: str = "192.168.1.100"
    mqtt_port: int = 1883
    can_interface: str = "can0"
    
    # Training
    local_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    
    # Data collection
    sequence_length: int = 20
    buffer_size: int = 1000
    
    # Privacy
    privacy_mechanism: str = "output_perturbation"
    noise_scale: float = 0.01
    gradient_clip_norm: float = 1.0


class DataCollector:
    """
    Collects data from CAN bus and device metrics.
    Maintains rolling buffer for training.
    """
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.buffer = deque(maxlen=config.buffer_size)
        self.current_state = {}
        self.running = False
        
        # Feature names
        self.feature_names = [
            'speed_kmh', 'battery_level', 'throttle', 'brake', 
            'steering', 'gear', 'location_x', 'location_y',
            'speed_delta', 'throttle_brake_ratio', 'message_frequency',
            'inter_arrival_time', 'payload_entropy',
            'cpu_usage_percent', 'cpu_temperature_c', 'memory_used_mb',
            'memory_total_mb', 'memory_usage_percent', 'load_average_1min',
            'throttling_state', 'network_rx_bytes', 'network_tx_bytes',
            'can_rx_count', 'can_tx_count', 'can_error_count'
        ]
        
        # For computing derived features
        self.last_values = {}
        self.message_times = deque(maxlen=100)
        
    def start(self):
        """Start data collection threads"""
        self.running = True
        
        # CAN listener thread
        self.can_thread = threading.Thread(target=self._can_listener, daemon=True)
        self.can_thread.start()
        
        # Device metrics thread
        self.metrics_thread = threading.Thread(target=self._metrics_collector, daemon=True)
        self.metrics_thread.start()
        
        print(f"Data collector started for {self.config.can_interface}")
    
    def stop(self):
        """Stop data collection"""
        self.running = False
    
    def _can_listener(self):
        """Listen for CAN messages"""
        try:
            import can
            bus = can.interface.Bus(channel=self.config.can_interface, 
                                   interface='socketcan')
            
            while self.running:
                msg = bus.recv(timeout=1.0)
                if msg:
                    self._process_can_message(msg)
                    
        except Exception as e:
            print(f"CAN listener error: {e}")
            # Fall back to simulated data
            self._simulate_can_data()
    
    def _simulate_can_data(self):
        """Simulate CAN data if real interface unavailable"""
        import random
        
        while self.running:
            # Simulate vehicle state
            self.current_state.update({
                'speed_kmh': max(0, self.current_state.get('speed_kmh', 50) + random.gauss(0, 2)),
                'battery_level': max(0, min(100, self.current_state.get('battery_level', 80) - 0.01)),
                'throttle': max(0, min(100, 30 + random.gauss(0, 10))),
                'brake': max(0, min(100, random.gauss(0, 5))),
                'steering': random.gauss(0, 5),
                'gear': random.randint(0, 5),
                'location_x': self.current_state.get('location_x', 0) + random.gauss(0, 1),
                'location_y': self.current_state.get('location_y', 0) + random.gauss(0, 1),
            })
            
            self._compute_derived_features()
            self._add_to_buffer()
            time.sleep(0.1)  # 10 Hz
    
    def _process_can_message(self, msg):
        """Process received CAN message"""
        can_id = msg.arbitration_id
        data = msg.data
        
        # Decode based on CAN ID
        if can_id == 0x123:  # Speed
            self.current_state['speed_kmh'] = struct.unpack('<f', data[:4])[0]
        elif can_id == 0x124:  # Battery
            self.current_state['battery_level'] = struct.unpack('<f', data[:4])[0]
        elif can_id == 0x125:  # Throttle
            self.current_state['throttle'] = struct.unpack('<f', data[:4])[0]
        elif can_id == 0x126:  # Brake
            self.current_state['brake'] = struct.unpack('<f', data[:4])[0]
        elif can_id == 0x127:  # Steering
            self.current_state['steering'] = struct.unpack('<f', data[:4])[0]
        elif can_id == 0x128:  # Gear
            self.current_state['gear'] = struct.unpack('<i', data[:4])[0]
        
        # Record message time
        self.message_times.append(time.time())
        
        self._compute_derived_features()
        self._add_to_buffer()
    
    def _compute_derived_features(self):
        """Compute derived features from raw data"""
        current_time = time.time()
        
        # Speed delta
        if 'speed_kmh' in self.last_values:
            self.current_state['speed_delta'] = (
                self.current_state.get('speed_kmh', 0) - self.last_values['speed_kmh']
            )
        else:
            self.current_state['speed_delta'] = 0
        
        # Throttle/brake ratio (suspicious if both high)
        throttle = self.current_state.get('throttle', 0)
        brake = self.current_state.get('brake', 0)
        self.current_state['throttle_brake_ratio'] = (
            throttle * brake / 100.0 if throttle + brake > 0 else 0
        )
        
        # Message frequency (messages per second)
        recent_times = [t for t in self.message_times if current_time - t < 1.0]
        self.current_state['message_frequency'] = len(recent_times)
        
        # Inter-arrival time
        if len(self.message_times) >= 2:
            self.current_state['inter_arrival_time'] = (
                self.message_times[-1] - self.message_times[-2]
            )
        else:
            self.current_state['inter_arrival_time'] = 0.1
        
        # Payload entropy (simplified)
        self.current_state['payload_entropy'] = np.random.uniform(0, 1)
        
        # Update last values
        self.last_values = self.current_state.copy()
    
    def _metrics_collector(self):
        """Collect device metrics"""
        while self.running:
            try:
                metrics = self._get_device_metrics()
                self.current_state.update(metrics)
            except Exception as e:
                print(f"Metrics collection error: {e}")
            time.sleep(0.5)  # 2 Hz for device metrics
    
    def _get_device_metrics(self) -> Dict[str, float]:
        """Get current device metrics"""
        metrics = {}
        
        try:
            # CPU usage
            with open('/proc/stat', 'r') as f:
                cpu_line = f.readline()
                cpu_times = [int(x) for x in cpu_line.split()[1:]]
                idle = cpu_times[3]
                total = sum(cpu_times)
                
                if hasattr(self, '_last_cpu_times'):
                    idle_delta = idle - self._last_cpu_times[0]
                    total_delta = total - self._last_cpu_times[1]
                    metrics['cpu_usage_percent'] = 100 * (1 - idle_delta / total_delta)
                else:
                    metrics['cpu_usage_percent'] = 0
                self._last_cpu_times = (idle, total)
            
            # CPU temperature
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                metrics['cpu_temperature_c'] = int(f.read()) / 1000.0
            
            # Memory
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    key, value = line.split(':')
                    meminfo[key] = int(value.strip().split()[0])
                metrics['memory_total_mb'] = meminfo['MemTotal'] / 1024
                metrics['memory_used_mb'] = (meminfo['MemTotal'] - meminfo['MemAvailable']) / 1024
                metrics['memory_usage_percent'] = metrics['memory_used_mb'] / metrics['memory_total_mb'] * 100
            
            # Load average
            with open('/proc/loadavg', 'r') as f:
                metrics['load_average_1min'] = float(f.read().split()[0])
            
            # Throttling (Pi specific)
            try:
                import subprocess
                result = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True)
                throttled = int(result.stdout.split('=')[1], 16)
                metrics['throttling_state'] = 1.0 if throttled != 0 else 0.0
            except:
                metrics['throttling_state'] = 0.0
            
            # Network stats
            with open('/proc/net/dev', 'r') as f:
                for line in f:
                    if 'eth0' in line or 'wlan0' in line:
                        parts = line.split()
                        metrics['network_rx_bytes'] = int(parts[1])
                        metrics['network_tx_bytes'] = int(parts[9])
                        break
            
            # CAN stats (if available)
            metrics['can_rx_count'] = getattr(self, '_can_rx_count', 0)
            metrics['can_tx_count'] = getattr(self, '_can_tx_count', 0)
            metrics['can_error_count'] = getattr(self, '_can_error_count', 0)
            
        except Exception as e:
            # Return defaults if reading fails
            metrics = {
                'cpu_usage_percent': 25.0,
                'cpu_temperature_c': 45.0,
                'memory_used_mb': 512.0,
                'memory_total_mb': 2048.0,
                'memory_usage_percent': 25.0,
                'load_average_1min': 0.5,
                'throttling_state': 0.0,
                'network_rx_bytes': 0,
                'network_tx_bytes': 0,
                'can_rx_count': 0,
                'can_tx_count': 0,
                'can_error_count': 0,
            }
        
        return metrics
    
    def _add_to_buffer(self):
        """Add current state to buffer"""
        # Create feature vector in consistent order
        features = []
        for name in self.feature_names:
            features.append(self.current_state.get(name, 0.0))
        
        self.buffer.append(np.array(features, dtype=np.float32))
    
    def get_training_data(self, sequence_length: int) -> Optional[torch.Tensor]:
        """Get data for training as sequences"""
        if len(self.buffer) < sequence_length * 2:
            return None
        
        # Convert buffer to array
        data = np.array(list(self.buffer))
        
        # Create sequences
        num_sequences = len(data) - sequence_length + 1
        sequences = np.array([
            data[i:i+sequence_length] 
            for i in range(num_sequences)
        ])
        
        return torch.from_numpy(sequences)
    
    def get_num_features(self) -> int:
        """Get number of features"""
        return len(self.feature_names)


class FederatedClient:
    """
    Federated learning client for edge devices.
    Handles local training, privacy, and server communication.
    """
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.client_id = config.client_id
        
        # Initialize components
        self.data_collector = DataCollector(config)
        self.model = None
        self.optimizer = None
        
        # Privacy mechanism
        self.privacy = self._init_privacy()
        
        # MQTT client
        self.mqtt_client = None
        self.connected = False
        
        # Training state
        self.current_round = 0
        self.training_history = []
        
        # Message queues
        self.model_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Status
        self.status = "initialized"
        self.last_training_loss = 0.0
        self.last_training_time = 0.0
        
    def _init_privacy(self):
        """Initialize privacy mechanism"""
        if self.config.privacy_mechanism == "input_perturbation":
            return InputPerturbation(
                noise_scale=self.config.noise_scale,
                clip_norm=self.config.gradient_clip_norm
            )
        elif self.config.privacy_mechanism == "output_perturbation":
            return OutputPerturbation(
                noise_scale=self.config.noise_scale,
                clip_norm=self.config.gradient_clip_norm
            )
        else:
            return None
    
    def initialize_model(self, model_state: Optional[Dict] = None):
        """Initialize or update local model"""
        num_features = self.data_collector.get_num_features()
        
        self.model = EdgeAutoencoder(
            input_dim=num_features,
            hidden_dim=32,
            latent_dim=8,
            num_layers=1,
            seq_length=self.config.sequence_length,
            model_type="gru"
        )
        
        if model_state:
            self.model.load_state_dict(model_state)
            print(f"[{self.client_id}] Loaded global model")
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        print(f"[{self.client_id}] Model initialized with {self.model.get_num_parameters()} parameters")
    
    def connect_mqtt(self):
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
        
        print(f"[{self.client_id}] Connecting to MQTT broker {self.config.mqtt_broker}...")
        self.mqtt_client.connect(self.config.mqtt_broker, self.config.mqtt_port)
        self.mqtt_client.loop_start()
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connect callback"""
        if hasattr(rc, 'value'):
            rc = rc.value
        
        if rc == 0:
            print(f"[{self.client_id}] Connected to MQTT broker")
            self.connected = True
            
            # Subscribe to topics
            client.subscribe(f"federated/model/global")
            client.subscribe(f"federated/aggregation/trigger")
            client.subscribe(f"federated/clients/{self.client_id}/commands")
            
            # Announce presence
            self._publish_status("online")
        else:
            print(f"[{self.client_id}] MQTT connection failed: {rc}")
    
    def _on_message(self, client, userdata, message):
        """MQTT message callback"""
        topic = message.topic
        
        try:
            if topic == "federated/model/global":
                # Received global model
                model_data = pickle.loads(message.payload)
                self.model_queue.put(model_data)
                print(f"[{self.client_id}] Received global model (round {model_data.get('round', '?')})")
                
            elif topic == "federated/aggregation/trigger":
                # Training trigger
                trigger = json.loads(message.payload)
                self.command_queue.put(('train', trigger))
                print(f"[{self.client_id}] Received training trigger")
                
            elif f"clients/{self.client_id}" in topic:
                # Client-specific command
                command = json.loads(message.payload)
                self.command_queue.put(('command', command))
                
        except Exception as e:
            print(f"[{self.client_id}] Error processing message: {e}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """MQTT disconnect callback"""
        self.connected = False
        print(f"[{self.client_id}] Disconnected from MQTT broker")
    
    def _publish_status(self, status: str):
        """Publish client status"""
        if self.mqtt_client and self.connected:
            payload = json.dumps({
                'client_id': self.client_id,
                'status': status,
                'timestamp': time.time(),
                'round': self.current_round,
                'last_loss': self.last_training_loss,
            })
            self.mqtt_client.publish("federated/clients/status", payload)
    
    def local_train(self) -> Tuple[OrderedDict, Dict]:
        """
        Perform local training on edge data.
        
        Returns:
            model_update: Model state dict or gradient update
            metrics: Training metrics
        """
        if self.model is None:
            print(f"[{self.client_id}] Model not initialized")
            return None, {}
        
        self.status = "training"
        self._publish_status("training")
        
        start_time = time.time()
        
        # Get training data
        data = self.data_collector.get_training_data(self.config.sequence_length)
        if data is None or len(data) < self.config.batch_size:
            print(f"[{self.client_id}] Insufficient data for training")
            return None, {'error': 'insufficient_data'}
        
        # Apply input perturbation if configured
        if isinstance(self.privacy, InputPerturbation):
            data = self.privacy.perturb(data)
        
        # Create DataLoader
        dataset = TensorDataset(data, data)  # Autoencoder: input = target
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Store initial weights for computing update
        initial_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstructed, latent = self.model(batch_x)
                loss = nn.MSELoss()(reconstructed, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / max(num_batches, 1)
        training_time = time.time() - start_time
        
        # Compute model update (difference from initial)
        model_update = OrderedDict()
        for name, param in self.model.state_dict().items():
            model_update[name] = param - initial_weights[name]
        
        # Apply output perturbation if configured
        if isinstance(self.privacy, OutputPerturbation):
            model_update = self.privacy.perturb_model_update(model_update)
        
        # Metrics
        metrics = {
            'loss': avg_loss,
            'training_time': training_time,
            'num_samples': len(data),
            'num_batches': num_batches,
            'privacy_mechanism': self.config.privacy_mechanism,
        }
        
        self.last_training_loss = avg_loss
        self.last_training_time = training_time
        self.training_history.append(metrics)
        
        print(f"[{self.client_id}] Training complete: loss={avg_loss:.4f}, time={training_time:.2f}s")
        
        self.status = "idle"
        self._publish_status("training_complete")
        
        return model_update, metrics
    
    def send_model_update(self, model_update: OrderedDict, metrics: Dict):
        """Send model update to server"""
        if not self.connected:
            print(f"[{self.client_id}] Not connected to broker")
            return
        
        payload = {
            'client_id': self.client_id,
            'round': self.current_round,
            'model_update': model_update,
            'metrics': metrics,
            'timestamp': time.time(),
        }
        
        # Serialize with pickle
        data = pickle.dumps(payload)
        
        self.mqtt_client.publish("federated/model/updates", data)
        print(f"[{self.client_id}] Sent model update ({len(data)} bytes)")
    
    def apply_global_model(self, model_data: Dict):
        """Apply received global model"""
        if 'model_state' in model_data:
            self.model.load_state_dict(model_data['model_state'])
            self.current_round = model_data.get('round', self.current_round + 1)
            print(f"[{self.client_id}] Applied global model (round {self.current_round})")
    
    def detect_anomaly(self, data: torch.Tensor) -> Tuple[bool, float]:
        """Run anomaly detection on data"""
        if self.model is None:
            return False, 0.0
        
        self.model.eval()
        with torch.no_grad():
            is_anomaly, score = self.model.detect_anomaly(data.unsqueeze(0))
        
        return is_anomaly.item(), score.item()
    
    def run(self):
        """Main client loop"""
        print(f"[{self.client_id}] Starting federated learning client...")
        
        # Start data collection
        self.data_collector.start()
        
        # Connect to MQTT
        self.connect_mqtt()
        
        # Wait for initial connection
        time.sleep(2)
        
        # Initialize model (will be updated when global model arrives)
        self.initialize_model()
        
        try:
            while True:
                # Check for global model updates
                try:
                    model_data = self.model_queue.get_nowait()
                    self.apply_global_model(model_data)
                except queue.Empty:
                    pass
                
                # Check for commands
                try:
                    cmd_type, cmd_data = self.command_queue.get_nowait()
                    if cmd_type == 'train':
                        # Perform local training
                        model_update, metrics = self.local_train()
                        if model_update:
                            self.send_model_update(model_update, metrics)
                except queue.Empty:
                    pass
                
                # Periodic anomaly detection on recent data
                data = self.data_collector.get_training_data(self.config.sequence_length)
                if data is not None and len(data) > 0:
                    is_anomaly, score = self.detect_anomaly(data[-1])
                    if is_anomaly:
                        print(f"[{self.client_id}] ⚠️ ANOMALY DETECTED: score={score:.4f}")
                        self._publish_alert(score)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"[{self.client_id}] Shutting down...")
        finally:
            self.data_collector.stop()
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
    
    def _publish_alert(self, score: float):
        """Publish anomaly alert"""
        if self.mqtt_client and self.connected:
            alert = {
                'client_id': self.client_id,
                'type': 'anomaly',
                'score': score,
                'timestamp': time.time(),
            }
            self.mqtt_client.publish("federated/alerts/attack", json.dumps(alert))


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Edge Client')
    parser.add_argument('--client-id', default='edge_1', help='Client identifier')
    parser.add_argument('--broker', default='192.168.1.100', help='MQTT broker IP')
    parser.add_argument('--port', type=int, default=1883, help='MQTT broker port')
    parser.add_argument('--can-interface', default='can0', help='CAN interface')
    parser.add_argument('--privacy', default='output_perturbation',
                       choices=['none', 'input_perturbation', 'output_perturbation'],
                       help='Privacy mechanism')
    
    args = parser.parse_args()
    
    config = ClientConfig(
        client_id=args.client_id,
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        can_interface=args.can_interface,
        privacy_mechanism=args.privacy,
    )
    
    client = FederatedClient(config)
    client.run()


if __name__ == "__main__":
    main()