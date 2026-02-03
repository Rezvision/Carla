# attack_generator.py - Synthetic CAN Attack Generator
"""
Generates synthetic CAN bus attacks based on CANtack patterns.
Reference: https://github.com/ascarecrowhat/CANtack

Attack Types:
1. DoS Flood - Overwhelm bus with high-frequency messages
2. Fuzzing - Random payload injection
3. Replay - Replay captured legitimate messages
4. Spoofing - Fake sensor values (gradual or sudden)
5. Suspension - Stop legitimate message transmission
6. Targeted ID - Attack specific CAN IDs
7. Gradual Drift - Slowly modify values to avoid detection
8. Burst Injection - Periodic high-intensity attacks
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass, field
from enum import Enum
import random
import time
from collections import deque


class AttackType(Enum):
    NORMAL = "normal"
    DOS_FLOOD = "dos_flood"
    FUZZING = "fuzzing"
    REPLAY = "replay"
    SPOOFING = "spoofing"
    SUSPENSION = "suspension"
    TARGETED_ID = "targeted_id"
    GRADUAL_DRIFT = "gradual_drift"
    BURST_INJECTION = "burst_injection"
    MIXED = "mixed"  # Combination of multiple attacks


@dataclass
class AttackConfig:
    """Configuration for attack generation"""
    
    # Attack probabilities (sum to 1 for random selection)
    attack_weights: Dict[str, float] = field(default_factory=lambda: {
        'dos_flood': 0.15,
        'fuzzing': 0.15,
        'replay': 0.10,
        'spoofing': 0.20,
        'suspension': 0.10,
        'targeted_id': 0.10,
        'gradual_drift': 0.10,
        'burst_injection': 0.10,
    })
    
    # DoS Flood parameters
    dos_frequency_multiplier: float = 10.0  # 10x normal frequency
    dos_duration_steps: int = 50  # Attack duration in time steps
    
    # Fuzzing parameters
    fuzzing_probability: float = 0.3  # Probability of fuzzing each field
    fuzzing_range: Tuple[float, float] = (-100.0, 100.0)  # Random value range
    
    # Replay parameters
    replay_buffer_size: int = 100  # Messages to buffer for replay
    replay_delay_steps: int = 10  # Delay before replaying
    
    # Spoofing parameters
    spoofing_targets: List[str] = field(default_factory=lambda: [
        'speed_kmh', 'battery_level', 'throttle', 'brake'
    ])
    spoofing_deviation: float = 0.5  # Max deviation factor from normal
    spoofing_sudden: bool = True  # Sudden vs gradual spoofing
    
    # Suspension parameters
    suspension_probability: float = 0.3  # Probability of dropping message
    suspension_target_ids: List[str] = field(default_factory=lambda: ['speed_kmh'])
    
    # Gradual drift parameters
    drift_rate: float = 0.02  # Change per time step (2% drift)
    drift_max_deviation: float = 0.3  # Maximum drift from original
    
    # Burst injection parameters
    burst_duration_steps: int = 5  # Short intense bursts
    burst_interval_steps: int = 50  # Time between bursts
    burst_intensity: float = 2.0  # Intensity multiplier
    
    # General
    attack_ratio: float = 0.2  # Ratio of attack to normal samples
    min_attack_duration: int = 10  # Minimum attack length
    max_attack_duration: int = 100  # Maximum attack length


@dataclass
class CANMessage:
    """Represents a CAN message with metadata"""
    timestamp: float
    can_id: int
    data: Dict[str, float]
    is_attack: bool = False
    attack_type: AttackType = AttackType.NORMAL
    attack_confidence: float = 0.0  # For labeling certainty


class VehicleState:
    """Simulates realistic vehicle state for normal data generation"""
    
    def __init__(self):
        self.speed = 0.0
        self.battery = 100.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        self.gear = 0
        self.location_x = 0.0
        self.location_y = 0.0
        
        # Driving behavior parameters
        self.target_speed = 0.0
        self.acceleration_rate = 0.5
        self.deceleration_rate = 0.8
        
        # State tracking
        self.time_step = 0
        self.driving_mode = "accelerating"  # accelerating, cruising, braking, stopped
        
    def update(self) -> Dict[str, float]:
        """Update vehicle state and return current values"""
        self.time_step += 1
        
        # Simulate driving behavior
        self._update_driving_mode()
        self._update_speed()
        self._update_controls()
        self._update_location()
        self._update_battery()
        
        return self.get_state()
    
    def _update_driving_mode(self):
        """Change driving mode periodically"""
        if self.time_step % 100 == 0:
            modes = ["accelerating", "cruising", "braking", "stopped"]
            weights = [0.3, 0.4, 0.2, 0.1]
            self.driving_mode = random.choices(modes, weights)[0]
            
            if self.driving_mode == "accelerating":
                self.target_speed = random.uniform(30, 120)
            elif self.driving_mode == "cruising":
                self.target_speed = self.speed
            elif self.driving_mode == "braking":
                self.target_speed = max(0, self.speed - random.uniform(10, 30))
            else:
                self.target_speed = 0
    
    def _update_speed(self):
        """Update speed towards target"""
        if self.speed < self.target_speed:
            self.speed = min(self.target_speed, 
                           self.speed + self.acceleration_rate + random.uniform(-0.1, 0.2))
        elif self.speed > self.target_speed:
            self.speed = max(self.target_speed,
                           self.speed - self.deceleration_rate + random.uniform(-0.1, 0.1))
        
        # Add noise
        self.speed = max(0, self.speed + random.gauss(0, 0.5))
    
    def _update_controls(self):
        """Update throttle, brake, steering"""
        if self.driving_mode == "accelerating":
            self.throttle = min(100, 30 + (self.target_speed - self.speed) * 2 + random.gauss(0, 5))
            self.brake = max(0, random.gauss(0, 2))
        elif self.driving_mode == "braking":
            self.throttle = max(0, random.gauss(0, 2))
            self.brake = min(100, 20 + (self.speed - self.target_speed) * 3 + random.gauss(0, 5))
        elif self.driving_mode == "cruising":
            self.throttle = max(0, 20 + random.gauss(0, 5))
            self.brake = max(0, random.gauss(0, 2))
        else:
            self.throttle = 0
            self.brake = max(0, 10 + random.gauss(0, 2))
        
        # Steering with smooth changes
        self.steering += random.gauss(0, 2)
        self.steering = np.clip(self.steering, -45, 45)
        
        # Gear based on speed
        if self.speed < 1:
            self.gear = 0
        elif self.speed < 20:
            self.gear = 1
        elif self.speed < 40:
            self.gear = 2
        elif self.speed < 60:
            self.gear = 3
        elif self.speed < 80:
            self.gear = 4
        else:
            self.gear = 5
    
    def _update_location(self):
        """Update simulated location"""
        # Simple movement based on speed and steering
        angle = np.radians(self.steering)
        speed_ms = self.speed / 3.6  # Convert to m/s
        
        self.location_x += speed_ms * np.cos(angle) * 0.1  # 100ms time step
        self.location_y += speed_ms * np.sin(angle) * 0.1
    
    def _update_battery(self):
        """Simulate battery drain"""
        # Drain based on throttle and speed
        drain_rate = 0.001 * (self.throttle / 100) * (1 + self.speed / 100)
        self.battery = max(0, self.battery - drain_rate + random.gauss(0, 0.0001))
    
    def get_state(self) -> Dict[str, float]:
        """Get current vehicle state as dictionary"""
        return {
            'speed_kmh': self.speed,
            'battery_level': self.battery,
            'throttle': self.throttle,
            'brake': self.brake,
            'steering': self.steering,
            'gear': float(self.gear),
            'location_x': self.location_x,
            'location_y': self.location_y,
        }


class CANAttackGenerator:
    """
    Generates synthetic CAN attacks for IDS training.
    Based on CANtack methodology.
    """
    
    def __init__(self, config: Optional[AttackConfig] = None):
        self.config = config or AttackConfig()
        self.vehicle = VehicleState()
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)
        
        # Attack state
        self.current_attack = AttackType.NORMAL
        self.attack_remaining_steps = 0
        self.drift_state = {}  # Track gradual drift values
        self.burst_counter = 0
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'attack_samples': 0,
            'attacks_by_type': {t.value: 0 for t in AttackType}
        }
    
    def generate_sample(self) -> Tuple[Dict[str, float], bool, str]:
        """
        Generate a single sample (normal or attack).
        
        Returns:
            data: Feature dictionary
            is_attack: Boolean attack label
            attack_type: String attack type
        """
        # Get normal vehicle state
        normal_state = self.vehicle.update()
        
        # Store in replay buffer
        self.replay_buffer.append(normal_state.copy())
        
        # Decide if this should be an attack
        if self.attack_remaining_steps > 0:
            # Continue current attack
            data, attack_type = self._apply_attack(normal_state)
            self.attack_remaining_steps -= 1
            is_attack = True
        elif random.random() < self.config.attack_ratio:
            # Start new attack
            self._start_new_attack()
            data, attack_type = self._apply_attack(normal_state)
            is_attack = True
        else:
            # Normal sample
            data = normal_state
            attack_type = AttackType.NORMAL.value
            is_attack = False
        
        # Update statistics
        self.stats['total_samples'] += 1
        if is_attack:
            self.stats['attack_samples'] += 1
            self.stats['attacks_by_type'][attack_type] += 1
        
        return data, is_attack, attack_type
    
    def _start_new_attack(self):
        """Start a new random attack"""
        # Select attack type based on weights
        attack_types = list(self.config.attack_weights.keys())
        weights = list(self.config.attack_weights.values())
        selected = random.choices(attack_types, weights)[0]
        
        self.current_attack = AttackType(selected)
        self.attack_remaining_steps = random.randint(
            self.config.min_attack_duration,
            self.config.max_attack_duration
        )
        
        # Initialize attack-specific state
        if self.current_attack == AttackType.GRADUAL_DRIFT:
            self.drift_state = {k: 0.0 for k in self.vehicle.get_state().keys()}
    
    def _apply_attack(self, normal_state: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """Apply current attack to normal state"""
        
        if self.current_attack == AttackType.DOS_FLOOD:
            return self._dos_flood_attack(normal_state)
        elif self.current_attack == AttackType.FUZZING:
            return self._fuzzing_attack(normal_state)
        elif self.current_attack == AttackType.REPLAY:
            return self._replay_attack(normal_state)
        elif self.current_attack == AttackType.SPOOFING:
            return self._spoofing_attack(normal_state)
        elif self.current_attack == AttackType.SUSPENSION:
            return self._suspension_attack(normal_state)
        elif self.current_attack == AttackType.TARGETED_ID:
            return self._targeted_id_attack(normal_state)
        elif self.current_attack == AttackType.GRADUAL_DRIFT:
            return self._gradual_drift_attack(normal_state)
        elif self.current_attack == AttackType.BURST_INJECTION:
            return self._burst_injection_attack(normal_state)
        else:
            return normal_state, AttackType.NORMAL.value
    
    def _dos_flood_attack(self, state: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        DoS Flood: Inject rapid high-frequency messages.
        Manifests as unusual message timing patterns.
        """
        data = state.copy()
        
        # Add timing anomaly marker (simulated via feature)
        # In real attack, this would be many messages in short time
        data['message_frequency'] = state.get('message_frequency', 10) * self.config.dos_frequency_multiplier
        
        # DoS often sends repeated or incrementing values
        if random.random() < 0.5:
            # Repeat last value
            pass
        else:
            # Slight random perturbation
            for key in ['speed_kmh', 'throttle', 'brake']:
                if key in data:
                    data[key] += random.uniform(-1, 1)
        
        return data, AttackType.DOS_FLOOD.value
    
    def _fuzzing_attack(self, state: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        Fuzzing: Inject random values into CAN fields.
        Tests system response to unexpected inputs.
        """
        data = state.copy()
        
        for key in data.keys():
            if random.random() < self.config.fuzzing_probability:
                # Random value in range
                data[key] = random.uniform(*self.config.fuzzing_range)
        
        return data, AttackType.FUZZING.value
    
    def _replay_attack(self, state: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        Replay: Re-send previously captured legitimate messages.
        Hard to detect as messages are valid.
        """
        if len(self.replay_buffer) >= self.config.replay_delay_steps:
            # Get old message
            replay_idx = random.randint(0, min(len(self.replay_buffer) - 1, 
                                               self.config.replay_delay_steps))
            data = list(self.replay_buffer)[replay_idx].copy()
        else:
            data = state.copy()
        
        return data, AttackType.REPLAY.value
    
    def _spoofing_attack(self, state: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        Spoofing: Inject fake sensor values.
        Can cause dangerous vehicle behavior.
        """
        data = state.copy()
        
        for target in self.config.spoofing_targets:
            if target in data:
                if self.config.spoofing_sudden:
                    # Sudden large change
                    deviation = data[target] * self.config.spoofing_deviation
                    data[target] += random.choice([-1, 1]) * deviation
                else:
                    # Gradually increasing deviation
                    deviation = data[target] * self.config.spoofing_deviation * 0.1
                    data[target] += deviation
        
        # Clamp to reasonable ranges
        if 'speed_kmh' in data:
            data['speed_kmh'] = max(0, data['speed_kmh'])
        if 'battery_level' in data:
            data['battery_level'] = np.clip(data['battery_level'], 0, 100)
        if 'throttle' in data:
            data['throttle'] = np.clip(data['throttle'], 0, 100)
        if 'brake' in data:
            data['brake'] = np.clip(data['brake'], 0, 100)
        
        return data, AttackType.SPOOFING.value
    
    def _suspension_attack(self, state: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        Suspension: Drop/suppress legitimate messages.
        Can cause loss of critical information.
        """
        data = state.copy()
        
        # Set suspended fields to zero or last known value
        for target in self.config.suspension_target_ids:
            if target in data and random.random() < self.config.suspension_probability:
                data[target] = 0.0  # Or could use NaN
        
        return data, AttackType.SUSPENSION.value
    
    def _targeted_id_attack(self, state: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        Targeted ID: Attack specific CAN message ID.
        Focused attack on critical signals.
        """
        data = state.copy()
        
        # Target speed specifically (critical for safety)
        if 'speed_kmh' in data:
            # Either max out or zero
            if random.random() < 0.5:
                data['speed_kmh'] = 0.0  # Fake stopped
            else:
                data['speed_kmh'] = 200.0  # Fake high speed
        
        return data, AttackType.TARGETED_ID.value
    
    def _gradual_drift_attack(self, state: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        Gradual Drift: Slowly modify values to avoid detection.
        Subtle attack that builds over time.
        """
        data = state.copy()
        
        for key in self.config.spoofing_targets:
            if key in data and key in self.drift_state:
                # Accumulate drift
                drift_direction = random.choice([-1, 1])
                self.drift_state[key] += drift_direction * self.config.drift_rate * abs(data[key] + 0.1)
                
                # Clamp drift
                max_drift = abs(data[key] + 0.1) * self.config.drift_max_deviation
                self.drift_state[key] = np.clip(self.drift_state[key], -max_drift, max_drift)
                
                # Apply drift
                data[key] += self.drift_state[key]
        
        return data, AttackType.GRADUAL_DRIFT.value
    
    def _burst_injection_attack(self, state: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        Burst Injection: Periodic intense attack bursts.
        Hard to detect due to intermittent nature.
        """
        data = state.copy()
        
        self.burst_counter += 1
        
        # Check if in burst period
        cycle_position = self.burst_counter % self.config.burst_interval_steps
        in_burst = cycle_position < self.config.burst_duration_steps
        
        if in_burst:
            # Apply intense modification
            for key in ['speed_kmh', 'throttle', 'brake']:
                if key in data:
                    data[key] *= self.config.burst_intensity
                    data[key] += random.uniform(-10, 10)
        
        return data, AttackType.BURST_INJECTION.value
    
    def generate_dataset(
        self, 
        num_samples: int,
        include_device_metrics: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate complete dataset for training.
        
        Args:
            num_samples: Number of samples to generate
            include_device_metrics: Whether to include device monitoring features
            
        Returns:
            X: Feature array (num_samples, num_features)
            y: Labels (num_samples,) - 0 for normal, 1 for attack
            attack_types: List of attack type strings
        """
        samples = []
        labels = []
        attack_types = []
        
        for _ in range(num_samples):
            data, is_attack, attack_type = self.generate_sample()
            
            # Add simulated device metrics
            if include_device_metrics:
                data.update(self._generate_device_metrics(is_attack))
            
            samples.append(list(data.values()))
            labels.append(1 if is_attack else 0)
            attack_types.append(attack_type)
        
        X = np.array(samples, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        
        return X, y, attack_types
    
    def _generate_device_metrics(self, is_attack: bool) -> Dict[str, float]:
        """Generate simulated device metrics"""
        base_cpu = 25.0
        base_temp = 45.0
        base_mem = 512.0
        
        # Attacks might cause higher resource usage
        attack_factor = 1.3 if is_attack else 1.0
        
        return {
            'cpu_usage_percent': base_cpu * attack_factor + random.gauss(0, 5),
            'cpu_temperature_c': base_temp * attack_factor + random.gauss(0, 2),
            'memory_used_mb': base_mem * attack_factor + random.gauss(0, 20),
            'memory_total_mb': 2048.0,
            'memory_usage_percent': (base_mem * attack_factor / 2048.0) * 100,
            'load_average_1min': 0.5 * attack_factor + random.gauss(0, 0.1),
            'throttling_state': 1.0 if (is_attack and random.random() < 0.1) else 0.0,
            'network_rx_bytes': 1000 * attack_factor + random.gauss(0, 100),
            'network_tx_bytes': 500 * attack_factor + random.gauss(0, 50),
            'can_rx_count': 100 * (attack_factor ** 2) + random.gauss(0, 10),
            'can_tx_count': 50 + random.gauss(0, 5),
            'can_error_count': 5 * (attack_factor - 1) + random.gauss(0, 1) if is_attack else random.gauss(0, 0.5),
        }
    
    def generate_sequences(
        self,
        num_sequences: int,
        sequence_length: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        """
        Generate sequences for autoencoder training.
        
        Returns:
            X: Tensor of shape (num_sequences, sequence_length, num_features)
            y: Tensor of shape (num_sequences,) - 1 if any attack in sequence
            attack_types: List of attack types per sequence
        """
        total_samples = num_sequences * sequence_length
        X_flat, y_flat, types_flat = self.generate_dataset(total_samples)
        
        num_features = X_flat.shape[1]
        
        # Reshape into sequences
        X = X_flat.reshape(num_sequences, sequence_length, num_features)
        
        # Sequence label: 1 if any attack in sequence
        y_reshaped = y_flat.reshape(num_sequences, sequence_length)
        y = (y_reshaped.sum(axis=1) > 0).astype(np.int64)
        
        # Attack types per sequence
        attack_types = []
        for i in range(num_sequences):
            start = i * sequence_length
            end = start + sequence_length
            attack_types.append(types_flat[start:end])
        
        return torch.from_numpy(X), torch.from_numpy(y), attack_types
    
    def get_statistics(self) -> Dict:
        """Get generation statistics"""
        total = self.stats['total_samples']
        if total == 0:
            return self.stats
        
        stats = self.stats.copy()
        stats['attack_ratio'] = self.stats['attack_samples'] / total
        stats['attack_distribution'] = {
            k: v / max(1, self.stats['attack_samples'])
            for k, v in self.stats['attacks_by_type'].items()
            if v > 0
        }
        return stats


def create_training_data(
    num_train: int = 10000,
    num_val: int = 2000,
    sequence_length: int = 20,
    attack_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Create training and validation datasets.
    
    Returns:
        Dictionary with X_train, y_train, X_val, y_val tensors
    """
    random.seed(seed)
    np.random.seed(seed)
    
    config = AttackConfig(attack_ratio=attack_ratio)
    generator = CANAttackGenerator(config)
    
    print(f"Generating {num_train} training sequences...")
    X_train, y_train, _ = generator.generate_sequences(num_train, sequence_length)
    
    print(f"Generating {num_val} validation sequences...")
    X_val, y_val, _ = generator.generate_sequences(num_val, sequence_length)
    
    stats = generator.get_statistics()
    print(f"\nGeneration Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Attack ratio: {stats['attack_ratio']:.2%}")
    print(f"  Attack distribution: {stats.get('attack_distribution', {})}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'num_features': X_train.shape[-1]
    }


if __name__ == "__main__":
    print("Testing CAN Attack Generator...")
    
    # Create generator
    config = AttackConfig(attack_ratio=0.3)
    generator = CANAttackGenerator(config)
    
    # Generate some samples
    print("\nGenerating samples...")
    for i in range(10):
        data, is_attack, attack_type = generator.generate_sample()
        status = f"ATTACK ({attack_type})" if is_attack else "NORMAL"
        print(f"Sample {i+1}: {status}")
        print(f"  Speed: {data['speed_kmh']:.1f} km/h, Throttle: {data['throttle']:.1f}%")
    
    # Generate dataset
    print("\nGenerating training dataset...")
    data = create_training_data(num_train=1000, num_val=200)
    
    print(f"\nDataset shapes:")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  y_train: {data['y_train'].shape}")
    print(f"  X_val: {data['X_val'].shape}")
    print(f"  y_val: {data['y_val'].shape}")
    print(f"  Attack rate (train): {data['y_train'].float().mean():.2%}")
    
    print("\n✓ Attack generator test passed!")
