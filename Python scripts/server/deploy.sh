#!/bin/bash
# deploy.sh - Deployment scripts for Federated Learning IDS
# Supports both edge devices (Raspberry Pi) and central server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================${NC}\n"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Detect system type
detect_system() {
    if [ -f /proc/device-tree/model ]; then
        if grep -q "Raspberry Pi" /proc/device-tree/model; then
            echo "raspberry_pi"
            return
        fi
    fi
    
    if [ "$(uname -m)" = "x86_64" ]; then
        echo "server"
    else
        echo "unknown"
    fi
}

# Setup for Raspberry Pi edge device
setup_edge() {
    print_header "Setting up Edge Device (Raspberry Pi)"
    
    # Update system
    echo "Updating system packages..."
    sudo apt-get update
    sudo apt-get upgrade -y
    
    # Install dependencies
    echo "Installing dependencies..."
    sudo apt-get install -y \
        python3-pip \
        python3-venv \
        can-utils \
        libatlas-base-dev \
        libopenblas-dev \
        mosquitto-clients
    
    # Create virtual environment
    echo "Creating Python virtual environment..."
    python3 -m venv fed_ids_env
    source fed_ids_env/bin/activate
    
    # Install Python packages
    echo "Installing Python packages..."
    pip install --upgrade pip
    pip install \
        torch==2.0.0 \
        numpy \
        paho-mqtt \
        python-can \
        scikit-learn \
        RPi.GPIO
    
    # Install TFLite runtime for Pi
    echo "Installing TFLite runtime..."
    pip install tflite-runtime
    
    # Setup CAN interface
    echo "Configuring CAN interface..."
    sudo modprobe can
    sudo modprobe can_raw
    sudo modprobe mcp251x
    
    # Add CAN to modules
    echo "can" | sudo tee -a /etc/modules
    echo "can_raw" | sudo tee -a /etc/modules
    
    # Create systemd service
    echo "Creating systemd service..."
    cat << 'EOF' | sudo tee /etc/systemd/system/fed-ids-edge.service
[Unit]
Description=Federated Learning IDS Edge Client
After=network.target can0.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/fed_ids
ExecStart=/home/pi/fed_ids/fed_ids_env/bin/python /home/pi/fed_ids/fed_client.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable fed-ids-edge
    
    # Create project directory
    mkdir -p /home/pi/fed_ids/models
    mkdir -p /home/pi/fed_ids/data
    mkdir -p /home/pi/fed_ids/logs
    
    print_header "Edge Setup Complete!"
    echo "Next steps:"
    echo "1. Copy fed_client.py and other Python files to /home/pi/fed_ids/"
    echo "2. Configure client ID in the script"
    echo "3. Set MQTT broker IP address"
    echo "4. Start service: sudo systemctl start fed-ids-edge"
}

# Setup for central server
setup_server() {
    print_header "Setting up Central Server"
    
    # Update system
    echo "Updating system packages..."
    sudo apt-get update
    sudo apt-get upgrade -y
    
    # Install dependencies
    echo "Installing dependencies..."
    sudo apt-get install -y \
        python3-pip \
        python3-venv \
        mosquitto \
        mosquitto-clients
    
    # Configure MQTT broker
    echo "Configuring MQTT broker..."
    cat << 'EOF' | sudo tee /etc/mosquitto/conf.d/federated.conf
# Federated Learning IDS MQTT Configuration
listener 1883 0.0.0.0
allow_anonymous true
max_connections -1
message_size_limit 0

# Logging
log_dest file /var/log/mosquitto/mosquitto.log
log_type all
EOF

    sudo systemctl restart mosquitto
    sudo systemctl enable mosquitto
    
    # Create virtual environment
    echo "Creating Python virtual environment..."
    python3 -m venv fed_ids_env
    source fed_ids_env/bin/activate
    
    # Install Python packages
    echo "Installing Python packages..."
    pip install --upgrade pip
    pip install \
        torch \
        numpy \
        paho-mqtt \
        scikit-learn \
        matplotlib \
        tensorboard
    
    # Create systemd service
    echo "Creating systemd service..."
    cat << 'EOF' | sudo tee /etc/systemd/system/fed-ids-server.service
[Unit]
Description=Federated Learning IDS Server
After=network.target mosquitto.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/fed_ids
ExecStart=/opt/fed_ids/fed_ids_env/bin/python /opt/fed_ids/fed_server.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable fed-ids-server
    
    # Create project directory
    sudo mkdir -p /opt/fed_ids/models
    sudo mkdir -p /opt/fed_ids/data
    sudo mkdir -p /opt/fed_ids/logs
    sudo chown -R $USER:$USER /opt/fed_ids
    
    print_header "Server Setup Complete!"
    echo "Next steps:"
    echo "1. Copy fed_server.py and other Python files to /opt/fed_ids/"
    echo "2. Configure server settings"
    echo "3. Start service: sudo systemctl start fed-ids-server"
    echo ""
    echo "MQTT broker is running on port 1883"
    echo "Server IP: $(hostname -I | awk '{print $1}')"
}

# Setup CAN interface
setup_can() {
    print_header "Setting up CAN Interface"
    
    local INTERFACE=${1:-can0}
    local BITRATE=${2:-1000000}
    
    echo "Configuring $INTERFACE at $BITRATE bps..."
    
    # Bring down interface if up
    sudo ip link set $INTERFACE down 2>/dev/null || true
    
    # Configure interface
    sudo ip link set $INTERFACE type can bitrate $BITRATE
    sudo ip link set $INTERFACE up
    
    # Verify
    if ip link show $INTERFACE | grep -q "UP"; then
        echo -e "${GREEN}✓ $INTERFACE is UP${NC}"
        ip -details link show $INTERFACE
    else
        print_error "$INTERFACE failed to come up"
        exit 1
    fi
}

# Test MQTT connectivity
test_mqtt() {
    print_header "Testing MQTT Connectivity"
    
    local BROKER=${1:-localhost}
    local PORT=${2:-1883}
    
    echo "Testing connection to $BROKER:$PORT..."
    
    # Subscribe in background
    timeout 5 mosquitto_sub -h $BROKER -p $PORT -t "test/connection" -C 1 &
    SUB_PID=$!
    
    sleep 1
    
    # Publish test message
    mosquitto_pub -h $BROKER -p $PORT -t "test/connection" -m "test_$(date +%s)"
    
    # Wait for subscriber
    wait $SUB_PID 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ MQTT connection successful${NC}"
    else
        print_error "MQTT connection failed"
        exit 1
    fi
}

# Generate training data
generate_data() {
    print_header "Generating Synthetic Training Data"
    
    local OUTPUT_DIR=${1:-./data}
    local NUM_SAMPLES=${2:-10000}
    
    echo "Generating $NUM_SAMPLES samples to $OUTPUT_DIR..."
    
    python3 << EOF
from attack_generator import create_training_data
import os

os.makedirs("$OUTPUT_DIR", exist_ok=True)

data = create_training_data(
    num_train=$NUM_SAMPLES,
    num_val=int($NUM_SAMPLES * 0.2),
    sequence_length=20,
    attack_ratio=0.2
)

import torch
torch.save(data, "$OUTPUT_DIR/training_data.pt")
print(f"Saved training data to $OUTPUT_DIR/training_data.pt")
EOF
}

# Run privacy comparison
compare_privacy() {
    print_header "Running Privacy Mechanism Comparison"
    
    python3 << 'EOF'
from privacy import PrivacyComparison, InputPerturbation, OutputPerturbation
import torch

print("Comparing privacy mechanisms...\n")

# Create sample gradients
gradients = {
    'layer1.weight': torch.randn(32, 64),
    'layer1.bias': torch.randn(64),
    'layer2.weight': torch.randn(64, 22),
    'layer2.bias': torch.randn(22),
}

comparison = PrivacyComparison()

# Test Input Perturbation
ip = InputPerturbation(noise_scale=0.1)
data = torch.randn(100, 22)
perturbed_data = ip.perturb(data)
print(f"Input Perturbation - Epsilon: {ip.compute_epsilon():.4f}")

# Test Output Perturbation
op = OutputPerturbation(noise_scale=0.01)
perturbed_grads = op.perturb_gradients(gradients)

metrics = comparison.evaluate_gradient_leakage_resistance(
    'output_perturbation',
    gradients,
    perturbed_grads
)
print(f"Output Perturbation - Leakage Resistance: {metrics['leakage_resistance_score']:.4f}")

print(comparison.generate_comparison_report())
EOF
}

# Main menu
show_menu() {
    echo ""
    echo "Federated Learning IDS Deployment Tool"
    echo "======================================="
    echo ""
    echo "1) Setup Edge Device (Raspberry Pi)"
    echo "2) Setup Central Server"
    echo "3) Setup CAN Interface"
    echo "4) Test MQTT Connectivity"
    echo "5) Generate Training Data"
    echo "6) Compare Privacy Mechanisms"
    echo "7) Auto-detect and Setup"
    echo "8) Exit"
    echo ""
    read -p "Select option: " choice
    
    case $choice in
        1) setup_edge ;;
        2) setup_server ;;
        3) 
            read -p "Interface (default: can0): " iface
            read -p "Bitrate (default: 1000000): " bitrate
            setup_can ${iface:-can0} ${bitrate:-1000000}
            ;;
        4)
            read -p "MQTT Broker (default: localhost): " broker
            test_mqtt ${broker:-localhost}
            ;;
        5)
            read -p "Output directory (default: ./data): " dir
            read -p "Number of samples (default: 10000): " samples
            generate_data ${dir:-./data} ${samples:-10000}
            ;;
        6) compare_privacy ;;
        7)
            SYSTEM=$(detect_system)
            echo "Detected system: $SYSTEM"
            if [ "$SYSTEM" = "raspberry_pi" ]; then
                setup_edge
            elif [ "$SYSTEM" = "server" ]; then
                setup_server
            else
                print_error "Unknown system type"
            fi
            ;;
        8) exit 0 ;;
        *) print_error "Invalid option" ;;
    esac
}

# Entry point
if [ $# -eq 0 ]; then
    while true; do
        show_menu
    done
else
    case $1 in
        edge) setup_edge ;;
        server) setup_server ;;
        can) setup_can $2 $3 ;;
        mqtt) test_mqtt $2 $3 ;;
        data) generate_data $2 $3 ;;
        privacy) compare_privacy ;;
        *) 
            echo "Usage: $0 {edge|server|can|mqtt|data|privacy}"
            exit 1
            ;;
    esac
fi