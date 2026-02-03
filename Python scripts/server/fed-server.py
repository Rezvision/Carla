# fed_server.py - Federated Learning Aggregation Server
"""
Central server for federated learning IDS.
Aggregates model updates from edge devices, maintains global model,
and orchestrates training rounds.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import threading
import pickle
import os
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
import paho.mqtt.client as mqtt

# Local imports
try:
    from server_model import ServerTransformerAutoencoder, create_server_model
    from edge_model import EdgeAutoencoder
    from privacy import OutputPerturbation
    from config import federated_config, server_model_config, privacy_config
except ImportError:
    print("Warning: Some modules not found.")


@dataclass
class ServerConfig:
    """Server configuration"""
    mqtt_broker: str = "0.0.0.0"  # Listen on all interfaces
    mqtt_port: int = 1883
    
    # Federated learning
    min_clients: int = 2
    max_clients: int = 10
    round_timeout: int = 300  # 5 minutes
    num_rounds: int = 100
    
    # Aggregation
    aggregation_strategy: str = "fedavg"  # fedavg, fedprox
    fedprox_mu: float = 0.01
    
    # Model
    model_save_dir: str = "./models"
    save_every_n_rounds: int = 10
    
    # Server-side training
    server_epochs: int = 5
    server_batch_size: int = 64
    server_learning_rate: float = 0.0001


@dataclass
class ClientInfo:
    """Track connected client information"""
    client_id: str
    status: str = "unknown"
    last_seen: float = 0.0
    rounds_participated: int = 0
    total_samples: int = 0
    avg_loss: float = 0.0


class ModelAggregator:
    """
    Aggregates model updates from multiple clients.
    Implements FedAvg and FedProx strategies.
    """
    
    def __init__(self, strategy: str = "fedavg", mu: float = 0.01):
        self.strategy = strategy
        self.mu = mu  # FedProx proximal term
        
    def aggregate(
        self,
        global_model: nn.Module,
        client_updates: List[Tuple[OrderedDict, Dict]],
        client_weights: Optional[List[float]] = None
    ) -> OrderedDict:
        """
        Aggregate client model updates.
        
        Args:
            global_model: Current global model
            client_updates: List of (model_update, metrics) from clients
            client_weights: Optional weights for each client (e.g., by sample count)
            
        Returns:
            Aggregated model state dict
        """
        if not client_updates:
            return global_model.state_dict()
        
        # Compute weights (default: equal weighting)
        if client_weights is None:
            total_samples = sum(m.get('num_samples', 1) for _, m in client_updates)
            client_weights = [m.get('num_samples', 1) / total_samples for _, m in client_updates]
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        if self.strategy == "fedavg":
            return self._fedavg(global_model, client_updates, client_weights)
        elif self.strategy == "fedprox":
            return self._fedprox(global_model, client_updates, client_weights)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")
    
    def _fedavg(
        self,
        global_model: nn.Module,
        client_updates: List[Tuple[OrderedDict, Dict]],
        weights: List[float]
    ) -> OrderedDict:
        """FedAvg: Weighted average of client updates"""
        global_state = global_model.state_dict()
        aggregated = OrderedDict()
        
        # Initialize with zeros
        for key in global_state.keys():
            aggregated[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        
        # Weighted sum of updates
        for (update, _), weight in zip(client_updates, weights):
            for key in aggregated.keys():
                if key in update:
                    # Add weighted update to global model
                    aggregated[key] += weight * (global_state[key] + update[key])
                else:
                    aggregated[key] += weight * global_state[key]
        
        return aggregated
    
    def _fedprox(
        self,
        global_model: nn.Module,
        client_updates: List[Tuple[OrderedDict, Dict]],
        weights: List[float]
    ) -> OrderedDict:
        """FedProx: FedAvg with proximal term regularization"""
        # First aggregate using FedAvg
        aggregated = self._fedavg(global_model, client_updates, weights)
        
        # Add proximal term (already applied on client side ideally)
        # Here we just use FedAvg result
        return aggregated


class FederatedServer:
    """
    Central federated learning server.
    Coordinates training across edge devices.
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        
        # Models
        self.global_model = None
        self.server_model = None  # Transformer model for server
        
        # Aggregator
        self.aggregator = ModelAggregator(
            strategy=config.aggregation_strategy,
            mu=config.fedprox_mu
        )
        
        # Client management
        self.clients: Dict[str, ClientInfo] = {}
        self.client_updates: Dict[str, Tuple[OrderedDict, Dict]] = {}
        
        # Round management
        self.current_round = 0
        self.round_start_time = 0.0
        self.round_in_progress = False
        
        # MQTT
        self.mqtt_client = None
        self.connected = False
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        
        # Statistics
        self.round_history = []
        
        # Create model directory
        os.makedirs(config.model_save_dir, exist_ok=True)
    
    def initialize_models(self, input_dim: int = 25):
        """Initialize global and server models"""
        
        # Edge model (for aggregation with clients)
        self.global_model = EdgeAutoencoder(
            input_dim=input_dim,
            hidden_dim=32,
            latent_dim=8,
            num_layers=1,
            seq_length=20,
            model_type="gru"
        )
        
        # Server transformer model (more powerful)
        self.server_model = ServerTransformerAutoencoder(
            input_dim=input_dim,
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=128,
            latent_dim=16,
            max_seq_len=100
        )
        
        print(f"Initialized models:")
        print(f"  Global (Edge) model: {self.global_model.get_num_parameters():,} parameters")
        print(f"  Server (Transformer) model: {self.server_model.get_num_parameters():,} parameters")
    
    def start_mqtt(self):
        """Start MQTT broker connection"""
        try:
            self.mqtt_client = mqtt.Client(
                client_id="fed_server",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
        except:
            self.mqtt_client = mqtt.Client(client_id="fed_server")
        
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.on_disconnect = self._on_disconnect
        
        print(f"Starting MQTT on {self.config.mqtt_broker}:{self.config.mqtt_port}")
        self.mqtt_client.connect(self.config.mqtt_broker, self.config.mqtt_port)
        self.mqtt_client.loop_start()
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connect callback"""
        if hasattr(rc, 'value'):
            rc = rc.value
            
        if rc == 0:
            print("Server connected to MQTT broker")
            self.connected = True
            
            # Subscribe to topics
            client.subscribe("federated/model/updates")
            client.subscribe("federated/clients/status")
            client.subscribe("federated/alerts/#")
        else:
            print(f"MQTT connection failed: {rc}")
    
    def _on_message(self, client, userdata, message):
        """MQTT message callback"""
        topic = message.topic
        
        try:
            if topic == "federated/model/updates":
                # Received client model update
                data = pickle.loads(message.payload)
                self._handle_client_update(data)
                
            elif topic == "federated/clients/status":
                # Client status update
                data = json.loads(message.payload)
                self._handle_client_status(data)
                
            elif "alerts" in topic:
                # Security alert
                data = json.loads(message.payload)
                self._handle_alert(data)
                
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """MQTT disconnect callback"""
        self.connected = False
        print("Server disconnected from MQTT broker")
    
    def _handle_client_update(self, data: Dict):
        """Handle received client model update"""
        client_id = data['client_id']
        model_update = data['model_update']
        metrics = data['metrics']
        round_num = data.get('round', self.current_round)
        
        with self.lock:
            # Only accept updates for current round
            if round_num == self.current_round and self.round_in_progress:
                self.client_updates[client_id] = (model_update, metrics)
                
                # Update client info
                if client_id in self.clients:
                    self.clients[client_id].rounds_participated += 1
                    self.clients[client_id].total_samples += metrics.get('num_samples', 0)
                    self.clients[client_id].avg_loss = metrics.get('loss', 0)
                
                print(f"Received update from {client_id}: "
                      f"loss={metrics.get('loss', 0):.4f}, "
                      f"samples={metrics.get('num_samples', 0)}")
                
                # Check if we have enough updates to aggregate
                if len(self.client_updates) >= self.config.min_clients:
                    self._trigger_aggregation()
    
    def _handle_client_status(self, data: Dict):
        """Handle client status update"""
        client_id = data['client_id']
        status = data['status']
        
        with self.lock:
            if client_id not in self.clients:
                self.clients[client_id] = ClientInfo(client_id=client_id)
                print(f"New client registered: {client_id}")
            
            self.clients[client_id].status = status
            self.clients[client_id].last_seen = time.time()
    
    def _handle_alert(self, data: Dict):
        """Handle security alert from client"""
        client_id = data.get('client_id', 'unknown')
        alert_type = data.get('type', 'unknown')
        score = data.get('score', 0)
        
        print(f"🚨 ALERT from {client_id}: {alert_type} (score={score:.4f})")
        
        # Could trigger additional actions here:
        # - Notify administrators
        # - Adjust model training
        # - Log to database
    
    def start_round(self):
        """Start a new federated learning round"""
        self.current_round += 1
        self.round_start_time = time.time()
        self.round_in_progress = True
        self.client_updates.clear()
        
        print(f"\n{'='*50}")
        print(f"Starting Round {self.current_round}")
        print(f"{'='*50}")
        print(f"Active clients: {len([c for c in self.clients.values() if c.status == 'online'])}")
        
        # Broadcast global model to clients
        self._broadcast_global_model()
        
        # Trigger training on clients
        self._trigger_client_training()
    
    def _broadcast_global_model(self):
        """Send global model to all clients"""
        if not self.connected:
            print("Not connected to broker")
            return
        
        payload = {
            'round': self.current_round,
            'model_state': self.global_model.state_dict(),
            'timestamp': time.time(),
        }
        
        data = pickle.dumps(payload)
        self.mqtt_client.publish("federated/model/global", data)
        print(f"Broadcast global model ({len(data)} bytes)")
    
    def _trigger_client_training(self):
        """Trigger training on all clients"""
        if not self.connected:
            return
        
        trigger = {
            'round': self.current_round,
            'timestamp': time.time(),
            'timeout': self.config.round_timeout,
        }
        
        self.mqtt_client.publish("federated/aggregation/trigger", json.dumps(trigger))
        print("Training trigger sent to clients")
    
    def _trigger_aggregation(self):
        """Perform model aggregation"""
        if not self.round_in_progress:
            return
        
        with self.lock:
            if len(self.client_updates) < self.config.min_clients:
                return
            
            print(f"\nAggregating {len(self.client_updates)} client updates...")
            
            # Collect updates
            updates = list(self.client_updates.values())
            
            # Aggregate
            aggregated_state = self.aggregator.aggregate(
                self.global_model,
                updates
            )
            
            # Update global model
            self.global_model.load_state_dict(aggregated_state)
            
            # Compute round statistics
            avg_loss = np.mean([m.get('loss', 0) for _, m in updates])
            total_samples = sum(m.get('num_samples', 0) for _, m in updates)
            
            round_stats = {
                'round': self.current_round,
                'num_clients': len(updates),
                'avg_loss': avg_loss,
                'total_samples': total_samples,
                'duration': time.time() - self.round_start_time,
            }
            self.round_history.append(round_stats)
            
            print(f"Round {self.current_round} complete:")
            print(f"  Clients: {len(updates)}")
            print(f"  Avg loss: {avg_loss:.4f}")
            print(f"  Total samples: {total_samples}")
            print(f"  Duration: {round_stats['duration']:.1f}s")
            
            # Save model periodically
            if self.current_round % self.config.save_every_n_rounds == 0:
                self._save_model()
            
            self.round_in_progress = False
    
    def _save_model(self):
        """Save current model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save global model
        global_path = os.path.join(
            self.config.model_save_dir,
            f"global_model_round{self.current_round}_{timestamp}.pt"
        )
        torch.save({
            'round': self.current_round,
            'model_state_dict': self.global_model.state_dict(),
            'round_history': self.round_history,
        }, global_path)
        
        # Save server model
        server_path = os.path.join(
            self.config.model_save_dir,
            f"server_model_round{self.current_round}_{timestamp}.pt"
        )
        torch.save({
            'round': self.current_round,
            'model_state_dict': self.server_model.state_dict(),
        }, server_path)
        
        print(f"Models saved: {global_path}")
    
    def server_side_training(self, aggregated_data: torch.Tensor):
        """
        Additional training on server with transformer model.
        Uses aggregated knowledge from all clients.
        """
        if self.server_model is None:
            return
        
        print("Performing server-side transformer training...")
        
        optimizer = optim.Adam(
            self.server_model.parameters(),
            lr=self.config.server_learning_rate
        )
        
        self.server_model.train()
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(aggregated_data, aggregated_data)
        loader = DataLoader(dataset, batch_size=self.config.server_batch_size, shuffle=True)
        
        for epoch in range(self.config.server_epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                reconstructed, _ = self.server_model(batch_x)
                loss = nn.MSELoss()(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"  Server epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
    
    def run(self):
        """Main server loop"""
        print("="*60)
        print("FEDERATED LEARNING IDS SERVER")
        print("="*60)
        
        # Initialize models
        self.initialize_models()
        
        # Start MQTT
        self.start_mqtt()
        
        # Wait for connection
        time.sleep(2)
        
        self.running = True
        
        try:
            while self.running and self.current_round < self.config.num_rounds:
                # Check for active clients
                active_clients = [
                    c for c in self.clients.values()
                    if c.status == "online" and time.time() - c.last_seen < 60
                ]
                
                if len(active_clients) >= self.config.min_clients:
                    if not self.round_in_progress:
                        # Start new round
                        self.start_round()
                    elif time.time() - self.round_start_time > self.config.round_timeout:
                        # Round timeout - aggregate what we have
                        print(f"Round {self.current_round} timeout - aggregating available updates")
                        self._trigger_aggregation()
                else:
                    if not self.round_in_progress:
                        print(f"Waiting for clients... ({len(active_clients)}/{self.config.min_clients})")
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self._save_model()
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total rounds: {self.current_round}")
        if self.round_history:
            final_loss = self.round_history[-1]['avg_loss']
            print(f"Final avg loss: {final_loss:.4f}")
    
    def get_status(self) -> Dict:
        """Get server status"""
        return {
            'current_round': self.current_round,
            'round_in_progress': self.round_in_progress,
            'num_clients': len(self.clients),
            'active_clients': len([c for c in self.clients.values() if c.status == "online"]),
            'updates_received': len(self.client_updates),
            'round_history': self.round_history[-10:],  # Last 10 rounds
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--broker', default='0.0.0.0', help='MQTT broker address')
    parser.add_argument('--port', type=int, default=1883, help='MQTT port')
    parser.add_argument('--min-clients', type=int, default=2, help='Minimum clients per round')
    parser.add_argument('--rounds', type=int, default=100, help='Number of training rounds')
    parser.add_argument('--model-dir', default='./models', help='Model save directory')
    
    args = parser.parse_args()
    
    config = ServerConfig(
        mqtt_broker=args.broker,
        mqtt_port=args.port,
        min_clients=args.min_clients,
        num_rounds=args.rounds,
        model_save_dir=args.model_dir,
    )
    
    server = FederatedServer(config)
    server.run()


if __name__ == "__main__":
    main()
