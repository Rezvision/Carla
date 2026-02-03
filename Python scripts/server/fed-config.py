# config.py - Federated Learning IDS Configuration
"""
Central configuration for the Federated Learning Intrusion Detection System.
Defines all hyperparameters, feature sets, and privacy settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import os


class PrivacyMechanism(Enum):
    """Privacy preservation mechanisms for federated learning"""
    NONE = "none"
    INPUT_PERTURBATION = "input_perturbation"      # Add noise to input data
    OUTPUT_PERTURBATION = "output_perturbation"    # Add noise to model updates
    DP_SGD = "dp_sgd"                              # Differentially private SGD
    GRADIENT_CLIPPING = "gradient_clipping"        # Clip gradients before sharing


class AggregationStrategy(Enum):
    """Federated aggregation strategies"""
    FED_AVG = "fedavg"           # Simple weighted averaging
    FED_PROX = "fedprox"         # Proximal term for heterogeneous data
    FED_NOVA = "fednova"         # Normalized averaging


@dataclass
class CANFeatureConfig:
    """CAN bus signal configuration matching CARLA simulation"""
    
    # CAN message IDs (from your DBF file)
    CAN_IDS: Dict[str, int] = field(default_factory=lambda: {
        "speed": 0x123,          # CAN_speed - 291 decimal
        "battery": 0x124,        # battery_current_capacity - 292 decimal
        "throttle": 0x125,
        "brake": 0x126,
        "steering": 0x127,
        "gear": 0x128,
    })
    
    # Feature names for model input
    SIGNAL_FEATURES: List[str] = field(default_factory=lambda: [
        "speed_kmh",
        "battery_level",
        "throttle",
        "brake",
        "steering",
        "gear",
    ])
    
    # Location features from CARLA (x, y coordinates)
    LOCATION_FEATURES: List[str] = field(default_factory=lambda: [
        "location_x",
        "location_y",
    ])
    
    # Derived CAN features for anomaly detection
    DERIVED_FEATURES: List[str] = field(default_factory=lambda: [
        "speed_delta",           # Rate of change
        "throttle_brake_ratio",  # Suspicious if both high
        "message_frequency",     # Messages per second per ID
        "inter_arrival_time",    # Time between messages
        "payload_entropy",       # Randomness in data bytes
    ])


@dataclass
class DeviceMetricsConfig:
    """Raspberry Pi device monitoring features"""
    
    METRICS: List[str] = field(default_factory=lambda: [
        "cpu_usage_percent",
        "cpu_temperature_c",
        "memory_used_mb",
        "memory_total_mb",
        "memory_usage_percent",
        "load_average_1min",
        "throttling_state",      # 0=normal, 1=throttled
        "network_rx_bytes",
        "network_tx_bytes",
        "can_rx_count",          # CAN messages received
        "can_tx_count",          # CAN messages transmitted
        "can_error_count",       # CAN bus errors
    ])
    
    # Sampling interval for device metrics (seconds)
    SAMPLING_INTERVAL: float = 0.1  # 100ms to match CAN message rate


@dataclass
class EdgeModelConfig:
    """GRU/LSTM Autoencoder configuration for edge devices (Raspberry Pi)"""
    
    # Model architecture
    MODEL_TYPE: str = "gru"     # "gru" or "lstm" - GRU is lighter
    INPUT_DIM: int = 22         # Total features (CAN + location + device)
    HIDDEN_DIM: int = 32        # Reduced for Pi's limited memory
    LATENT_DIM: int = 8         # Compressed representation
    NUM_LAYERS: int = 1         # Single layer for efficiency
    DROPOUT: float = 0.1
    BIDIRECTIONAL: bool = False  # Unidirectional for real-time
    
    # Sequence parameters
    SEQUENCE_LENGTH: int = 20   # Time steps in window
    STRIDE: int = 5             # Overlap between windows
    
    # Training parameters
    BATCH_SIZE: int = 16        # Small batch for limited RAM
    LEARNING_RATE: float = 0.001
    LOCAL_EPOCHS: int = 3       # Epochs per federated round
    
    # Anomaly detection threshold
    RECONSTRUCTION_THRESHOLD: float = 0.5  # MSE threshold for anomaly


@dataclass
class ServerModelConfig:
    """Transformer Autoencoder configuration for central server"""
    
    # Model architecture
    INPUT_DIM: int = 22         # Same feature space as edge
    D_MODEL: int = 64           # Transformer dimension
    NHEAD: int = 4              # Attention heads
    NUM_ENCODER_LAYERS: int = 3
    NUM_DECODER_LAYERS: int = 3
    DIM_FEEDFORWARD: int = 128
    DROPOUT: float = 0.1
    LATENT_DIM: int = 16        # Larger latent space for global patterns
    
    # Sequence parameters
    SEQUENCE_LENGTH: int = 50   # Longer sequences for pattern detection
    MAX_SEQUENCE_LENGTH: int = 100
    
    # Training parameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.0001
    AGGREGATION_EPOCHS: int = 5  # Training after aggregation


@dataclass
class PrivacyConfig:
    """Differential privacy and gradient protection settings"""
    
    # Active mechanism (will compare all three)
    ACTIVE_MECHANISM: PrivacyMechanism = PrivacyMechanism.OUTPUT_PERTURBATION
    
    # Input perturbation settings
    INPUT_NOISE_SCALE: float = 0.1      # Gaussian noise std dev
    INPUT_CLIP_NORM: float = 1.0        # Clip input values
    
    # Output perturbation settings (for model updates)
    OUTPUT_NOISE_SCALE: float = 0.01    # Noise added to gradients
    GRADIENT_CLIP_NORM: float = 1.0     # Max gradient norm
    
    # DP-SGD settings
    DP_EPSILON: float = 1.0             # Privacy budget
    DP_DELTA: float = 1e-5              # Privacy parameter
    DP_MAX_GRAD_NORM: float = 1.0       # Gradient clipping for DP
    DP_NOISE_MULTIPLIER: float = 1.1    # Noise multiplier
    
    # Secure aggregation
    SECURE_AGGREGATION: bool = True     # Use secure sum protocol
    MIN_CLIENTS_FOR_AGGREGATION: int = 2  # Minimum clients needed


@dataclass
class FederatedConfig:
    """Federated learning protocol configuration"""
    
    # Communication
    MQTT_BROKER: str = "192.168.1.100"  # On-premise server IP
    MQTT_PORT: int = 1883
    MQTT_KEEPALIVE: int = 60
    
    # Topics
    TOPIC_MODEL_GLOBAL: str = "federated/model/global"
    TOPIC_MODEL_UPDATE: str = "federated/model/updates"
    TOPIC_AGGREGATION_TRIGGER: str = "federated/aggregation/trigger"
    TOPIC_CLIENT_STATUS: str = "federated/clients/status"
    TOPIC_ATTACK_ALERT: str = "federated/alerts/attack"
    
    # Federated rounds
    NUM_ROUNDS: int = 100               # Total federated rounds
    ROUND_TIMEOUT: int = 300            # 5 minutes timeout per round
    MIN_CLIENTS_PER_ROUND: int = 2      # Minimum participating clients
    CLIENT_FRACTION: float = 1.0        # Fraction of clients per round
    
    # Aggregation
    AGGREGATION_STRATEGY: AggregationStrategy = AggregationStrategy.FED_AVG
    FED_PROX_MU: float = 0.01           # Proximal term coefficient
    
    # Model versioning
    MODEL_VERSION_FORMAT: str = "v{round}_{timestamp}"


@dataclass 
class AttackGeneratorConfig:
    """Synthetic attack generation based on CANtack patterns"""
    
    # Attack types to generate
    ATTACK_TYPES: List[str] = field(default_factory=lambda: [
        "dos_flood",             # Flood bus with messages
        "fuzzing",               # Random payload injection
        "replay",                # Replay captured messages
        "spoofing",              # Fake sensor values
        "suspension",            # Stop legitimate messages
        "targeted_id",           # Target specific CAN ID
        "gradual_drift",         # Slowly modify values
        "burst_injection",       # Periodic burst attacks
    ])
    
    # Attack parameters
    DOS_MESSAGES_PER_SECOND: int = 1000
    FUZZING_PROBABILITY: float = 0.3
    REPLAY_DELAY_MS: int = 100
    SPOOFING_DEVIATION: float = 0.5     # Max deviation from normal
    DRIFT_RATE: float = 0.01            # Value change per step
    BURST_DURATION_MS: int = 50
    BURST_INTERVAL_MS: int = 500
    
    # Attack injection ratio for training
    ATTACK_RATIO: float = 0.2           # 20% attack samples


@dataclass
class DataConfig:
    """Data collection and preprocessing settings"""
    
    # Data directories
    DATA_DIR: str = "./data"
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    MODEL_DIR: str = "./models"
    
    # Data collection
    BUFFER_SIZE: int = 10000            # Max samples in memory
    SAVE_INTERVAL: int = 1000           # Save every N samples
    
    # Preprocessing
    NORMALIZATION: str = "minmax"       # "minmax" or "standard"
    HANDLE_MISSING: str = "interpolate" # "interpolate", "zero", "drop"
    
    # Train/validation split
    VALIDATION_SPLIT: float = 0.2
    
    # Feature engineering
    ADD_TIME_FEATURES: bool = True      # Hour, minute, etc.
    ADD_ROLLING_STATS: bool = True      # Rolling mean, std
    ROLLING_WINDOW: int = 10


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    
    LOG_DIR: str = "./logs"
    LOG_LEVEL: str = "INFO"
    
    # Metrics to track
    TRACK_METRICS: List[str] = field(default_factory=lambda: [
        "reconstruction_loss",
        "anomaly_score",
        "detection_rate",
        "false_positive_rate",
        "communication_overhead",
        "privacy_budget_spent",
        "round_duration",
        "model_size_bytes",
    ])
    
    # Alerting
    ALERT_ON_ATTACK: bool = True
    ALERT_MQTT_TOPIC: str = "federated/alerts"


# Create global config instances
can_config = CANFeatureConfig()
device_config = DeviceMetricsConfig()
edge_model_config = EdgeModelConfig()
server_model_config = ServerModelConfig()
privacy_config = PrivacyConfig()
federated_config = FederatedConfig()
attack_config = AttackGeneratorConfig()
data_config = DataConfig()
logging_config = LoggingConfig()


def get_total_features() -> int:
    """Calculate total number of input features"""
    return (
        len(can_config.SIGNAL_FEATURES) +
        len(can_config.LOCATION_FEATURES) +
        len(can_config.DERIVED_FEATURES) +
        len(device_config.METRICS)
    )


def print_config_summary():
    """Print configuration summary"""
    print("\n" + "="*60)
    print("FEDERATED LEARNING IDS CONFIGURATION")
    print("="*60)
    print(f"\nFeatures:")
    print(f"  CAN signals:     {len(can_config.SIGNAL_FEATURES)}")
    print(f"  Location:        {len(can_config.LOCATION_FEATURES)}")
    print(f"  Derived CAN:     {len(can_config.DERIVED_FEATURES)}")
    print(f"  Device metrics:  {len(device_config.METRICS)}")
    print(f"  TOTAL:           {get_total_features()}")
    print(f"\nEdge Model (GRU Autoencoder):")
    print(f"  Hidden dim:      {edge_model_config.HIDDEN_DIM}")
    print(f"  Latent dim:      {edge_model_config.LATENT_DIM}")
    print(f"  Sequence length: {edge_model_config.SEQUENCE_LENGTH}")
    print(f"\nServer Model (Transformer Autoencoder):")
    print(f"  D_model:         {server_model_config.D_MODEL}")
    print(f"  Attention heads: {server_model_config.NHEAD}")
    print(f"  Encoder layers:  {server_model_config.NUM_ENCODER_LAYERS}")
    print(f"\nPrivacy:")
    print(f"  Mechanism:       {privacy_config.ACTIVE_MECHANISM.value}")
    print(f"  DP Epsilon:      {privacy_config.DP_EPSILON}")
    print(f"\nFederated Learning:")
    print(f"  Aggregation:     {federated_config.AGGREGATION_STRATEGY.value}")
    print(f"  Rounds:          {federated_config.NUM_ROUNDS}")
    print(f"  Min clients:     {federated_config.MIN_CLIENTS_PER_ROUND}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config_summary()
