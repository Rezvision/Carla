# Federated Learning Intrusion Detection System for Vehicle Networks

A cutting-edge federated learning system for detecting intrusions in vehicle CAN bus networks using distributed edge devices (Raspberry Pi) and a central aggregation server.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ON-PREMISE CENTRAL SERVER                        │
│                                                                      │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │           Transformer Autoencoder (Global Model)           │    │
│   │   • Aggregates knowledge from all edge devices             │    │
│   │   • Detects sophisticated cross-vehicle attack patterns    │    │
│   │   • Parameters: 64-dim, 4 heads, 3 encoder/decoder layers │    │
│   └───────────────────────────────────────────────────────────┘    │
│                                                                      │
│   Privacy: Input Perturbation | Output Perturbation | DP-SGD        │
│   Communication: MQTT (port 1883)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                ▲
                                │ MQTT (Encrypted model updates)
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  EDGE 1 (Pi)  │      │  EDGE 2 (Pi)  │      │  EDGE 3 (Pi)  │
│               │      │               │      │               │
│ GRU Autoenc.  │      │ GRU Autoenc.  │      │ GRU Autoenc.  │
│ 32-dim hidden │      │ 32-dim hidden │      │ 32-dim hidden │
│ 8-dim latent  │      │ 8-dim latent  │      │ 8-dim latent  │
│               │      │               │      │               │
│ Features:     │      │ Features:     │      │ Features:     │
│ • CAN signals │      │ • CAN signals │      │ • CAN signals │
│ • Location    │      │ • Location    │      │ • Location    │
│ • Device logs │      │ • Device logs │      │ • Device logs │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   [CAN Bus]               [CAN Bus]               [CAN Bus]
```

## 📁 Project Structure

```
federated_ids/
├── config.py              # All configuration parameters
├── edge_model.py          # GRU/LSTM Autoencoder for Raspberry Pi
├── server_model.py        # Transformer Autoencoder for server
├── privacy.py             # Differential privacy mechanisms
├── attack_generator.py    # CANtack-style synthetic attacks
├── fed_client.py          # Edge device federated client
├── fed_server.py          # Central aggregation server
├── data_processor.py      # Feature extraction & preprocessing
├── train.py               # Standalone training script
├── deploy.sh              # Deployment automation
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Server Setup (On-Premise)

```bash
# Run deployment script
chmod +x deploy.sh
./deploy.sh server

# Or manually:
pip install torch numpy paho-mqtt scikit-learn

# Start MQTT broker
sudo systemctl start mosquitto

# Start federation server
python fed_server.py --broker 0.0.0.0 --min-clients 2
```

### 2. Edge Device Setup (Raspberry Pi)

```bash
# Run deployment script
./deploy.sh edge

# Or manually:
pip install torch numpy paho-mqtt python-can RPi.GPIO tflite-runtime

# Setup CAN interface
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# Start client (replace with your server IP)
python fed_client.py --client-id edge_1 --broker 192.168.1.100
```

### 3. Pre-train Models (Optional)

```bash
# Train edge model with output perturbation privacy
python train.py --model edge --privacy output_perturbation --epochs 50

# Compare privacy mechanisms
python train.py --compare-privacy
```

## 📊 Features

### Input Features (25 total)

| Category | Features | Count |
|----------|----------|-------|
| **CAN Signals** | speed, battery, throttle, brake, steering, gear | 6 |
| **Location** | x, y coordinates from CARLA | 2 |
| **Derived CAN** | speed_delta, throttle_brake_ratio, msg_frequency, inter_arrival_time, payload_entropy | 5 |
| **Device Metrics** | cpu_usage, cpu_temp, memory, load_avg, throttling, network_rx/tx, can_rx/tx/error | 12 |

### Attack Types (CANtack-based)

| Attack | Description | Detection Method |
|--------|-------------|------------------|
| **DoS Flood** | High-frequency message injection | Message frequency anomaly |
| **Fuzzing** | Random payload injection | Value range anomaly |
| **Replay** | Re-send captured messages | Sequence pattern anomaly |
| **Spoofing** | Fake sensor values | Value deviation anomaly |
| **Suspension** | Drop legitimate messages | Missing message detection |
| **Gradual Drift** | Slowly modify values | Trend analysis |
| **Burst Injection** | Periodic intense attacks | Burst pattern detection |

## 🔐 Privacy Mechanisms

### Comparison Summary

| Mechanism | Privacy | Utility | Gradient Leakage Resistance | Overhead |
|-----------|---------|---------|----------------------------|----------|
| **Input Perturbation** | Medium | High | Medium | Low |
| **Output Perturbation** | High | Medium-High | High | Low |
| **DP-SGD** | Highest | Medium | Highest | High |

### Recommended Configuration

For Vehicle IDS, we recommend **Output Perturbation**:
- Good balance of privacy and utility
- Low computational overhead (suitable for Pi)
- Strong protection against gradient inversion attacks

```python
# In config.py
privacy_config.ACTIVE_MECHANISM = PrivacyMechanism.OUTPUT_PERTURBATION
privacy_config.OUTPUT_NOISE_SCALE = 0.01
privacy_config.GRADIENT_CLIP_NORM = 1.0
```

## 🔧 Configuration

### Edge Model (Raspberry Pi optimized)

```python
edge_model_config = EdgeModelConfig(
    MODEL_TYPE = "gru",        # Lighter than LSTM
    INPUT_DIM = 25,
    HIDDEN_DIM = 32,           # Small for Pi's memory
    LATENT_DIM = 8,
    NUM_LAYERS = 1,
    SEQUENCE_LENGTH = 20,
    BATCH_SIZE = 16,
)
```

### Server Model (More powerful)

```python
server_model_config = ServerModelConfig(
    INPUT_DIM = 25,
    D_MODEL = 64,
    NHEAD = 4,
    NUM_ENCODER_LAYERS = 3,
    NUM_DECODER_LAYERS = 3,
    LATENT_DIM = 16,
    SEQUENCE_LENGTH = 50,
)
```

### Federated Learning

```python
federated_config = FederatedConfig(
    MQTT_BROKER = "192.168.1.100",
    MIN_CLIENTS_PER_ROUND = 2,
    NUM_ROUNDS = 100,
    AGGREGATION_STRATEGY = "fedavg",
)
```

## 📡 MQTT Topics

| Topic | Direction | Purpose |
|-------|-----------|---------|
| `federated/model/global` | Server → Clients | Broadcast global model |
| `federated/model/updates` | Clients → Server | Send model updates |
| `federated/aggregation/trigger` | Server → Clients | Trigger training round |
| `federated/clients/status` | Clients → Server | Client status updates |
| `federated/alerts/attack` | Clients → Server | Attack detection alerts |

## 🧪 Testing

### Generate Synthetic Data

```bash
python attack_generator.py  # Creates sample attack data
```

### Test Privacy Mechanisms

```bash
python privacy.py  # Runs privacy comparison
```

### Test Models

```bash
python edge_model.py   # Test edge model
python server_model.py # Test server model
```

## 📈 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Detection Rate | > 90% | True positive rate for attacks |
| False Positive Rate | < 5% | False alarms on normal traffic |
| Latency | < 100ms | Real-time detection |
| Privacy (ε) | < 1.0 | Differential privacy budget |

## 🔗 Integration with Existing Code

This system integrates with your existing `vTCU_python_realcan_IDS_v7.py`:

```python
# In your TCU code, replace the VehicleIDS class with FederatedClient
from fed_client import FederatedClient, ClientConfig

config = ClientConfig(
    client_id="tcu_1",
    mqtt_broker="192.168.1.100",
    can_interface="can0",
)

fed_client = FederatedClient(config)
fed_client.run()
```

## 📚 References

- **CANtack**: Synthetic CAN attack generator - https://github.com/ascarecrowhat/CANtack
- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks"
- **Differential Privacy**: Abadi et al., "Deep Learning with Differential Privacy"
- **Transformer Autoencoders**: Vaswani et al., "Attention Is All You Need"

## 📝 License

MIT License - Feel free to use for research and commercial applications.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request
