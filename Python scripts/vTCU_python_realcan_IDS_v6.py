# vTCU_python_realcan_IDS_v7.py
import time
import json
import paho.mqtt.client as mqtt
import threading
import can
import socket
import struct
from datetime import datetime
import sys
import os
import ipaddress
import numpy as np
import pickle
from collections import deque, defaultdict
import signal
import atexit

# Try to import TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    print("WARNING: tflite_runtime not found. IDS will be disabled.")
    TFLITE_AVAILABLE = False

# Import GPIO
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("WARNING: RPi.GPIO not found. LED indicators will be disabled.")
    GPIO_AVAILABLE = False

# Try to import joblib for scaler loading
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    print("WARNING: joblib not found. Trying pickle instead.")
    JOBLIB_AVAILABLE = False

# LED GPIO pins (BCM numbering)
LED_RED = 17      # Attack detected (GPIO17 = Physical pin 11)
LED_GREEN = 27    # MQTT upload (GPIO27 = Physical pin 13)
LED_BLUE = 22     # System running (GPIO22 = Physical pin 15)
LED_YELLOW = 23   # IDS operational (GPIO23 = Physical pin 16)

class VehicleIDS:
    def __init__(self, model_path='vehicle_ids_model.tflite', 
                 scaler_path='scaler.pkl', 
                 config_path='model_config.pkl'):
        """Initialize the IDS with pre-trained TFLite model"""
        print("\n=== Initializing IDS System ===")
        
        # Check if all required files exist
        required_files = [model_path, scaler_path, config_path]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
            else:
                print(f"✓ Found: {file}")
        
        # Load configuration
        print("Loading configuration...")
        try:
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            self.window_size = self.config['window_size']
            self.feature_columns = self.config['feature_columns']
            print(f"✓ Config loaded: window_size={self.window_size}, features={len(self.feature_columns)}")
        except Exception as e:
            raise Exception(f"Failed to load config: {e}")
        
        # Load scaler - try joblib first, then pickle
        print("Loading scaler...")
        try:
            if JOBLIB_AVAILABLE:
                self.scaler = joblib.load(scaler_path)
                print("✓ Scaler loaded successfully with joblib")
            else:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✓ Scaler loaded successfully with pickle")
        except Exception as e:
            raise Exception(f"Failed to load scaler: {e}")
        
        # Load TFLite model
        print("Loading TFLite model...")
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✓ TFLite model loaded")
            print(f"  - Input shape: {self.input_details[0]['shape']}")
            print(f"  - Output shape: {self.output_details[0]['shape']}")
        except Exception as e:
            raise Exception(f"Failed to load TFLite model: {e}")
        
        # Initialize data buffer for complete vehicle states
        self.data_buffer = deque(maxlen=self.window_size)
        self.last_values = {}
        
        # Attack detection statistics
        self.detection_count = 0
        self.total_predictions = 0
        
        # Test the model with dummy data
        print("Testing IDS with dummy data...")
        try:
            self._test_inference()
            print("✓ IDS test successful")
        except Exception as e:
            raise Exception(f"IDS test failed: {e}")
        
        print("=== IDS System Initialized Successfully ===\n")
    
    def _test_inference(self):
        """Test the model with dummy data to ensure it's working"""
        # Create dummy data
        dummy_data = {col: 0.0 for col in self.feature_columns}
        
        # Fill buffer with dummy data
        for _ in range(self.window_size):
            self.data_buffer.append(dummy_data.copy())
        
        # Create window features
        window_features = []
        for data in self.data_buffer:
            features = self.create_features(data)
            window_features.append(features)
        
        # Prepare input
        X = np.array(window_features).reshape(1, self.window_size, -1)
        
        # Scale features
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape).astype(np.float32)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], X_scaled)
        self.interpreter.invoke()
        
        # Get prediction
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        
        # Clear buffer after test
        self.data_buffer.clear()
        
        return prediction
    
    def create_features(self, current_data):
        """Create features including deltas and rolling statistics"""
        features = []
        
        # Get current values and calculate deltas
        for col in self.feature_columns:
            if col in current_data:
                current_val = current_data[col]
                
                # Ensure numeric value
                try:
                    current_val = float(current_val)
                except:
                    current_val = 0.0
                
                # Original value
                features.append(current_val)
                
                # Delta
                if col in self.last_values:
                    delta = current_val - self.last_values[col]
                else:
                    delta = 0
                features.append(delta)
                
                # Update last value
                self.last_values[col] = current_val
                
                # Rolling statistics
                buffer_values = [d.get(col, current_val) for d in self.data_buffer]
                if len(buffer_values) > 0:
                    buffer_values.append(current_val)
                    
                    # Ensure all values are numeric
                    buffer_values = [float(v) if v is not None else 0.0 for v in buffer_values]
                    
                    rolling_mean = np.mean(buffer_values)
                    rolling_std = np.std(buffer_values) if len(buffer_values) > 1 else 0
                else:
                    rolling_mean = current_val
                    rolling_std = 0
                
                features.append(rolling_mean)
                features.append(rolling_std)
            else:
                # If feature missing, use zeros
                features.extend([0, 0, 0, 0])
                
        return np.array(features, dtype=np.float32)
    
    def predict(self, vehicle_data):
        """Predict if current data indicates an attack"""
        # Add current data to buffer
        self.data_buffer.append(vehicle_data.copy())
        
        # Need full window for prediction
        if len(self.data_buffer) < self.window_size:
            return False, 0.0
        
        try:
            # Create window features
            window_features = []
            for data in self.data_buffer:
                features = self.create_features(data)
                window_features.append(features)
            
            # Prepare input
            X = np.array(window_features).reshape(1, self.window_size, -1)
            
            # Scale features
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(original_shape).astype(np.float32)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], X_scaled)
            self.interpreter.invoke()
            
            # Get prediction
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
            
            # Update statistics
            self.total_predictions += 1
            
            # Lower threshold to 50% for more sensitive detection
            is_attack = prediction > 0.5
            
            if is_attack:
                self.detection_count += 1
            elif prediction > 0.3:  # Notable but not attack level
                print(f"📊 IDS: Elevated risk detected - {prediction*100:.2f}% (below 50% threshold)")
            
            # Also check for anomalous patterns in the data
            anomaly_detected = self._check_anomalies(vehicle_data)
            
            # Combine ML prediction with anomaly detection
            final_attack = is_attack or anomaly_detected
            
            # Boost confidence if anomaly detected
            if anomaly_detected and prediction > 0.3:
                prediction = min(prediction + 0.2, 1.0)
            
            return final_attack, float(prediction)
            
        except Exception as e:
            print(f"IDS prediction error: {e}")
            return False, 0.0
    
    def _check_anomalies(self, data):
        """Check for anomalous patterns in the data"""
        anomalies = []
        
        # Check for unrealistic values
        if 'Speed (km/h)' in data:
            if data['Speed (km/h)'] > 300:
                anomalies.append(f"Speed too high: {data['Speed (km/h)']} km/h")
            elif data['Speed (km/h)'] < -50:
                anomalies.append(f"Speed negative: {data['Speed (km/h)']} km/h")
        
        if 'Battery Level (%)' in data:
            if data['Battery Level (%)'] > 100:
                anomalies.append(f"Battery > 100%: {data['Battery Level (%)']}")
            elif data['Battery Level (%)'] < 0:
                anomalies.append(f"Battery < 0%: {data['Battery Level (%)']}")
        
        if 'Throttle' in data:
            if data['Throttle'] > 100:
                anomalies.append(f"Throttle > 100%: {data['Throttle']}")
            elif data['Throttle'] < 0:
                anomalies.append(f"Throttle < 0%: {data['Throttle']}")
        
        if 'Brake' in data:
            if data['Brake'] > 100:
                anomalies.append(f"Brake > 100%: {data['Brake']}")
            elif data['Brake'] < 0:
                anomalies.append(f"Brake < 0%: {data['Brake']}")
        
        # Check for impossible combinations
        if 'Throttle' in data and 'Brake' in data:
            if data['Throttle'] > 80 and data['Brake'] > 80:
                anomalies.append(f"Both throttle ({data['Throttle']}) and brake ({data['Brake']}) high")
        
        # Check for rapid changes (if we have history)
        if len(self.data_buffer) > 0:
            last_data = self.data_buffer[-1]
            
            # Check for impossibly rapid speed changes
            if 'Speed (km/h)' in data and 'Speed (km/h)' in last_data:
                speed_change = abs(data['Speed (km/h)'] - last_data['Speed (km/h)'])
                if speed_change > 50:  # More than 50 km/h change in 100ms is suspicious
                    anomalies.append(f"Rapid speed change: {speed_change:.1f} km/h in 100ms")
            
            # Check for rapid battery changes
            if 'Battery Level (%)' in data and 'Battery Level (%)' in last_data:
                battery_change = abs(data['Battery Level (%)'] - last_data['Battery Level (%)'])
                if battery_change > 10:  # More than 10% battery change in 100ms is impossible
                    anomalies.append(f"Rapid battery change: {battery_change:.1f}% in 100ms")
        
        if anomalies:
            print(f"⚠️  Anomalies detected: {', '.join(anomalies)}")
            return True
        
        return False

class TCU:
    def __init__(self, can_interface='can0', mqtt_broker_ip='192.168.1.100', mqtt_port=1883):
        """Initialize the TCU with IDS and LED support"""
        print("\n=== Starting TCU Initialization ===")
        
        # Initialize control flags FIRST
        self.running = True
        self.UDPrunning = True
        self.MQTTrunning = True
        
        # Initialize frequency-based detection
        self.message_timestamps = defaultdict(list)  # Track message timestamps by ID
        self.frequency_window = 1.0  # 1 second window
        self.frequency_threshold = 20  # More than 20 messages/second is suspicious
        self.frequency_attack_threshold = 50  # More than 50 messages/second is definitely an attack
        
        # LED control variables
        self.red_led_state = False  # Track LED state
        
        # Initialize GPIO for LEDs
        self.leds_initialized = False
        self.gpio_available = GPIO_AVAILABLE
        
        if self.gpio_available:
            try:
                # Suppress GPIO warnings
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                print("GPIO mode set to BCM (warnings suppressed)")
                
                # Setup pins with explicit error checking
                pins = [
                    (LED_RED, "Red/Attack", 17),
                    (LED_GREEN, "Green/MQTT", 27),
                    (LED_BLUE, "Blue/System", 22),
                    (LED_YELLOW, "Yellow/IDS", 23)
                ]
                
                for pin, name, bcm in pins:
                    try:
                        GPIO.setup(pin, GPIO.OUT)
                        GPIO.output(pin, GPIO.LOW)
                        print(f"✓ {name} LED configured on GPIO{bcm} (pin {pin})")
                    except Exception as e:
                        print(f"✗ Failed to setup {name} LED on GPIO{bcm}: {e}")
                
                # Turn on blue LED to indicate system is starting
                GPIO.output(LED_BLUE, GPIO.HIGH)
                self.leds_initialized = True
                print("✓ GPIO initialized - Blue LED ON (System starting)")
                
                # Test red LED with simple on/off
                print("Testing red LED (simple on/off)...")
                GPIO.output(LED_RED, GPIO.HIGH)
                print("  Red LED should be ON now")
                time.sleep(1.0)
                GPIO.output(LED_RED, GPIO.LOW)
                print("  Red LED should be OFF now")
                time.sleep(0.5)
                
                # Test red LED with PWM effect
                print("Testing red LED (PWM flash)...")
                for i in range(3):
                    GPIO.output(LED_RED, GPIO.HIGH)
                    time.sleep(0.2)
                    GPIO.output(LED_RED, GPIO.LOW)
                    time.sleep(0.2)
                print("✓ Red LED test complete")
                
                # Register cleanup functions
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                atexit.register(self._cleanup_gpio)
            except Exception as e:
                print(f"⚠ GPIO initialization failed: {e}")
                self.gpio_available = False
                self.leds_initialized = False
        else:
            print("⚠ GPIO not available - LEDs disabled")
        
        # Initialize IDS
        self.ids_enabled = False
        if TFLITE_AVAILABLE:
            try:
                print("\nInitializing IDS module...")
                self.ids = VehicleIDS()
                self.ids_enabled = True
                
                # Turn on yellow LED to indicate IDS is operational
                if self.gpio_available and self.leds_initialized:
                    GPIO.output(LED_YELLOW, GPIO.HIGH)
                    print("✓ IDS operational - Yellow LED ON")
                else:
                    print("✓ IDS operational")
                    
            except Exception as e:
                print(f"\n✗ ERROR: Could not initialize IDS: {e}")
                print("Running without IDS functionality")
                
                # Keep yellow LED off to indicate IDS is not operational
                if self.gpio_available and self.leds_initialized:
                    GPIO.output(LED_YELLOW, GPIO.LOW)
        else:
            print("\n⚠ TFLite not available - IDS disabled")
            if self.gpio_available and self.leds_initialized:
                GPIO.output(LED_YELLOW, GPIO.LOW)
        
        # Store current vehicle data and last known values
        self.current_vehicle_data = {}
        self.last_known_values = {
            "Speed (km/h)": 0.0,
            "Battery Level (%)": 85.0,
            "Throttle": 0.0,
            "Brake": 0.0,
            "Steering": 0.0,
            "Gear": 0
        }
        
        # Track complete vehicle states for IDS
        self.last_complete_state_time = time.time()
        self.state_update_interval = 0.1  # Update IDS every 100ms with complete state
        
        # Store location separately since it's not from CAN
        self.last_location = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # Attack detection counters
        self.attack_count = 0
        self.frequency_attack_count = 0
        self.total_messages = 0
        self.ml_attack_detected = False
        self.ml_attack_confidence = 0.0
        
        # Track ML-based attack detection
        self.ml_attack_detected = False
        self.ml_attack_confidence = 0.0
        
        # Test IDS periodically
        if self.ids_enabled:
            self.ids_test_thread = threading.Thread(target=self._periodic_ids_test, daemon=True)
            self.ids_test_thread.start()
        
        # Rest of initialization
        print("\nInitializing CAN and network components...")
        
        self.last_sent_data = {}
        self.can_interface = can_interface
        self.mqtt_broker_ip = mqtt_broker_ip
        self.mqtt_port = mqtt_port
        
        # Set up physical CAN bus
        try:
            self.can_bus = can.interface.Bus(channel=self.can_interface, 
                                           interface='socketcan',
                                           bitrate=1000000)
            print(f"✓ Connected to CAN interface: {self.can_interface}")
        except Exception as e:
            print(f"✗ Error initializing CAN interface {self.can_interface}: {e}")
            self._shutdown_error()
            sys.exit(1)
            
        self.remote_brake = False
        self.SIGNATURE_BIT_MASK = 0x80
        
        # Start CAN listener thread
        self.listener_thread = threading.Thread(target=self._listen_to_can, daemon=True)
        self.listener_thread.start()
        print("✓ CAN listener started")
        
        # Start UDP listener thread
        self.UDPlistener_thread = threading.Thread(target=self.udp_listener, daemon=True)
        self.UDPlistener_thread.start()
        print("✓ UDP listener started")
        
        # Start periodic IDS evaluation thread
        self.ids_eval_thread = threading.Thread(target=self._periodic_ids_evaluation, daemon=True)
        self.ids_eval_thread.start()
        print("✓ Periodic IDS evaluation started")
        
        # Start periodic status thread
        self.status_thread = threading.Thread(target=self._periodic_status, daemon=True)
        self.status_thread.start()
        print("✓ Periodic status reporting started")
        
        # Set up MQTT client
        try:
            try:
                self.mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                                             client_id=f"TCU_{self.can_interface}")
            except AttributeError:
                self.mqtt_client = mqtt.Client(client_id=f"TCU_{self.can_interface}")
                
            self.mqtt_client.on_message = self._on_mqtt_command
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            print(f"Connecting to MQTT broker at {self.mqtt_broker_ip}:{self.mqtt_port}...")
            self.mqtt_client.connect(self.mqtt_broker_ip, self.mqtt_port, keepalive=60)
            self.mqtt_client.subscribe("vehicle/control")
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"✗ Error connecting to MQTT broker: {e}")
            self._shutdown_error()
            sys.exit(1)
        
        print("\n=== TCU Initialization Complete ===")
        print("\nLED Status Indicators:")
        print("  🔵 Blue: System running")
        print("  🟢 Green: MQTT data uploaded") 
        print("  🔴 Red: Attack detected (ML-based or frequency)")
        print("  🟡 Yellow: IDS operational")
        print(f"\nIDS Status: {'ENABLED ✓' if self.ids_enabled else 'DISABLED ✗'}")
        print("\nAttack Detection Methods:")
        print("  1. ML/AI-based anomaly detection (50% threshold)")
        print("     - Detects attacks based on learned data patterns")
        print("     - Independent of signature bits")
        print("  2. Frequency-based detection (>20 msg/sec suspicious, >50 msg/sec attack)")
        print("     - Detects flooding/DoS attacks")
        print("     - Red LED stays on for 3 seconds when detected")
        print("\nDashboard Notification:")
        print("  - Signature bit (0x80) used to notify dashboard only")
        print("  - Not used for ML-based IDS detection")
        print("\n")

    def _check_message_frequency(self, can_id):
        """Check if message frequency indicates an attack"""
        current_time = time.time()
        
        # Clean old timestamps
        self.message_timestamps[can_id] = [
            ts for ts in self.message_timestamps[can_id] 
            if current_time - ts < self.frequency_window
        ]
        
        # Add current timestamp
        self.message_timestamps[can_id].append(current_time)
        
        # Calculate frequency
        frequency = len(self.message_timestamps[can_id])
        
        # Debug every 10 messages
        if self.total_messages % 10 == 0:
            max_freq = max([len(timestamps) for timestamps in self.message_timestamps.values()]) if self.message_timestamps else 0
            print(f"DEBUG: Max frequency across all CAN IDs: {max_freq} msg/sec")
        
        # Check thresholds
        if frequency > self.frequency_attack_threshold:
            print(f"🚨 FREQUENCY ATTACK: CAN ID {hex(can_id)} - {frequency} messages/second!")
            print(f"   GPIO available: {self.gpio_available}, LEDs initialized: {self.leds_initialized}")
            return True, frequency
        elif frequency > self.frequency_threshold:
            print(f"⚠️  High frequency detected: CAN ID {hex(can_id)} - {frequency} messages/second")
            return False, frequency
        
        return False, frequency

    def _periodic_status(self):
        """Print periodic status including LED states"""
        while self.running:
            time.sleep(10)  # Every 10 seconds
            if self.frequency_attack_count > 0 or (self.ids_enabled and self.ids.detection_count > 0):
                print(f"\n📊 Attack Detection Status:")
                print(f"   Frequency attacks detected: {self.frequency_attack_count}")
                if self.ids_enabled:
                    print(f"   ML attacks detected: {self.ids.detection_count}")
                print(f"   Red LED state: {'ON' if self.red_led_state else 'OFF'}")
                print(f"   Total messages processed: {self.total_messages}\n")

    def _signal_handler(self, signum, frame):
        """Handle signals for clean shutdown"""
        print("\nReceived signal, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _cleanup_gpio(self):
        """Cleanup GPIO on exit"""
        if self.gpio_available and self.leds_initialized:
            try:
                GPIO.output(LED_RED, GPIO.LOW)
                GPIO.output(LED_GREEN, GPIO.LOW)
                GPIO.output(LED_BLUE, GPIO.LOW)
                GPIO.output(LED_YELLOW, GPIO.LOW)
                GPIO.cleanup()
                print("GPIO cleaned up")
            except:
                pass

    def _periodic_ids_test(self):
        """Periodically test IDS to ensure it's still working"""
        while self.running and self.ids_enabled:
            time.sleep(30)  # Test every 30 seconds
            try:
                # Simple test to ensure IDS is responsive
                if hasattr(self, 'ids') and self.ids:
                    # Just checking if we can access the model
                    _ = self.ids.window_size
                else:
                    # IDS lost, turn off yellow LED
                    if self.gpio_available and self.leds_initialized:
                        GPIO.output(LED_YELLOW, GPIO.LOW)
                    self.ids_enabled = False
                    print("⚠ IDS check failed - IDS disabled")
            except Exception as e:
                if self.gpio_available and self.leds_initialized:
                    GPIO.output(LED_YELLOW, GPIO.LOW)
                self.ids_enabled = False
                print(f"⚠ IDS check failed: {e}")

    def _shutdown_error(self):
        """Shutdown LEDs on error"""
        if self.gpio_available and self.leds_initialized:
            GPIO.output(LED_RED, GPIO.LOW)
            GPIO.output(LED_GREEN, GPIO.LOW)
            GPIO.output(LED_BLUE, GPIO.LOW)
            GPIO.output(LED_YELLOW, GPIO.LOW)
            GPIO.cleanup()

    def _flash_led(self, led_pin, duration):
        """Flash an LED for specified duration (for green LED only)"""
        if self.gpio_available and self.leds_initialized and led_pin == LED_GREEN:
            GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(led_pin, GPIO.LOW)

    def _listen_to_can(self):
        """Listen for CAN messages"""
        while self.running:
            try:
                msg = self.can_bus.recv(timeout=1.0)
                if msg is not None:
                    self.total_messages += 1
                    can_timestamp = msg.timestamp
                    timestamp_str = datetime.fromtimestamp(can_timestamp).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                    
                    # Check frequency-based attack
                    freq_attack, frequency = self._check_message_frequency(msg.arbitration_id)
                    
                    # Check signature bit (only for dashboard notification, not IDS)
                    has_signature_bit = self.is_message_attacked(msg.data)
                    
                    # Print status
                    status_parts = []
                    if has_signature_bit:
                        status_parts.append("SIGNATURE BIT SET")
                    if freq_attack:
                        status_parts.append(f"FREQ ATTACK: {frequency} msg/s")
                    
                    status = " - ".join(status_parts) if status_parts else "NORMAL"
                    print(f"[{timestamp_str}] Raw CAN Message Received: {msg} - Status: {status}")
                    
                    # Turn on red LED for frequency attacks
                    if freq_attack:
                        self.attack_count += 1
                        self.frequency_attack_count += 1
                        print(f"🚨 FREQUENCY ATTACK DETECTED #{self.frequency_attack_count} - Turning on RED LED")
                        print(f"   Current LED state: {self.red_led_state}")
                        
                        # Simple approach - just turn it on
                        if self.gpio_available and self.leds_initialized:
                            try:
                                # Read current state first
                                current_state = GPIO.input(LED_RED)
                                print(f"   Red LED current state before: {current_state}")
                                
                                GPIO.output(LED_RED, GPIO.HIGH)
                                self.red_led_state = True
                                
                                # Verify it's actually on
                                new_state = GPIO.input(LED_RED)
                                print(f"   Red LED state after: {new_state} (should be 1/HIGH)")
                                
                                if new_state == GPIO.HIGH:
                                    print("   ✓ Red LED is now ON")
                                else:
                                    print("   ✗ Red LED did not turn on!")
                                
                                # Start a timer to turn it off after 3 seconds
                                def delayed_off():
                                    time.sleep(3.0)
                                    if self.gpio_available and self.leds_initialized:
                                        GPIO.output(LED_RED, GPIO.LOW)
                                        self.red_led_state = False
                                        print("   ✓ Red LED turned OFF after 3 seconds")
                                
                                threading.Thread(target=delayed_off, daemon=True).start()
                                
                            except Exception as e:
                                print(f"   ✗ Failed to control red LED: {e}")
                                import traceback
                                traceback.print_exc()
                    else:
                        # Turn off LED if no attack and it's on
                        if self.red_led_state and not self.ml_attack_detected:
                            if self.gpio_available and self.leds_initialized:
                                try:
                                    GPIO.output(LED_RED, GPIO.LOW)
                                    self.red_led_state = False
                                except:
                                    pass
                    
                    self._process_can_message(msg, can_timestamp, has_signature_bit, freq_attack)
            except Exception as e:
                if self.running:
                    print(f"Error in CAN listener: {e}")

    def _process_can_message(self, msg, timestamp, is_attacked, freq_attack=False):
        """Process CAN messages and run IDS"""
        try:
            attribute_name = None
            decoded_value = None
            
            # Decode message based on ID - Extract actual value without signature bit
            if msg.arbitration_id == 0x123:  # Speed
                attribute_name = "Speed (km/h)"
                decoded_value = self._decode_float(msg)
            elif msg.arbitration_id == 0x124:  # Battery level
                attribute_name = "Battery Level (%)"
                decoded_value = self._decode_float(msg)
            elif msg.arbitration_id == 0x125:  # Throttle
                attribute_name = "Throttle"
                decoded_value = self._decode_float(msg)
            elif msg.arbitration_id == 0x126:  # Brake
                attribute_name = "Brake"
                decoded_value = self._decode_float(msg)
            elif msg.arbitration_id == 0x127:  # Steering
                attribute_name = "Steering"
                decoded_value = self._decode_float(msg)
            elif msg.arbitration_id == 0x128:  # Gear
                attribute_name = "Gear"
                decoded_value = self._decode_int(msg)
            
            # Update current vehicle data only if we got a valid value
            if attribute_name and decoded_value is not None:
                self.current_vehicle_data[attribute_name] = decoded_value
                # Also update last known value
                self.last_known_values[attribute_name] = decoded_value
                
                # Print what we decoded
                print(f"Decoded {attribute_name}: {decoded_value}")
            
            # Original processing continues
            if attribute_name and decoded_value is not None:
                # For dashboard notification, include signature bit
                # But IDS detection is separate and based only on data patterns
                dashboard_attack = is_attacked or freq_attack
                attack_info = ""
                if is_attacked:
                    attack_info = " (SIGNATURE BIT SET - Dashboard Notified)"
                if freq_attack:
                    attack_info += " (FREQUENCY ATTACK)"
                
                print(f"Processing {attribute_name}: {decoded_value}{attack_info}")
                
                timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                
                can_data = {
                    attribute_name: {
                        "value": decoded_value,
                        "timestamp": timestamp_str,
                        "raw_timestamp": timestamp,
                        "attacked": "yes" if dashboard_attack else "no"  # This is for dashboard only
                    }
                }
                self._update_aggregated_data(can_data)
                
                # Print attack statistics periodically
                if self.total_messages % 100 == 0:
                    ml_attack_rate = (self.ids.detection_count / self.ids.total_predictions * 100) if self.ids_enabled and self.ids.total_predictions > 0 else 0
                    print(f"\n📈 IDS ML Detection Statistics: {self.ids.detection_count}/{self.ids.total_predictions} attacks detected ({ml_attack_rate:.2f}%)")
                    print(f"📊 Frequency Attack Count: {self.attack_count} high-frequency bursts detected\n")
                
        except Exception as e:
            print(f"Error processing CAN message: {e}")
            import traceback
            traceback.print_exc()

    def _periodic_ids_evaluation(self):
        """Periodically evaluate complete vehicle state with IDS based on data patterns only"""
        while self.running and self.ids_enabled:
            time.sleep(self.state_update_interval)
            
            try:
                # Get complete vehicle data
                complete_data = self._ensure_complete_vehicle_data()
                
                # Only run IDS if we have meaningful data
                if self._validate_data_for_ids(complete_data):
                    # Run IDS prediction based purely on data patterns
                    # This is independent of signature bits - only looks at actual values
                    is_attack, confidence = self.ids.predict(complete_data)
                    
                    # Update ML attack status
                    self.ml_attack_detected = is_attack
                    self.ml_attack_confidence = confidence
                    
                    # Print periodic status
                    if self.ids.total_predictions % 10 == 0:  # Every 10th prediction
                        print(f"\n📊 ML-Based IDS Status (Data Pattern Analysis):")
                        print(f"   Total predictions: {self.ids.total_predictions}")
                        print(f"   Attacks detected: {self.ids.detection_count} ({self.ids.detection_count/self.ids.total_predictions*100:.1f}%)")
                        print(f"   Current confidence: {confidence*100:.2f}%")
                        print(f"   Attack detected: {'YES' if is_attack else 'NO'}")
                        print(f"   Current state: Speed={complete_data.get('Speed (km/h)', 0):.1f} km/h, "
                              f"Throttle={complete_data.get('Throttle', 0):.1f}%, "
                              f"Brake={complete_data.get('Brake', 0):.1f}%\n")
                    
                    # Handle attack detection
                    if is_attack:
                        print(f"\n🚨 [ML-IDS ALERT] Attack pattern detected with confidence: {confidence*100:.2f}%")
                        print(f"   This detection is based on learned data patterns, not signature bits")
                        print(f"   Vehicle state: Speed={complete_data.get('Speed (km/h)', 0):.1f} km/h, "
                              f"Throttle={complete_data.get('Throttle', 0):.1f}%, "
                              f"Brake={complete_data.get('Brake', 0):.1f}%, "
                              f"Battery={complete_data.get('Battery Level (%)', 0):.1f}%\n")
                        
                        # Turn on red LED for ML-detected attacks
                        print(f"🚨 ML ATTACK DETECTED - Turning on RED LED")
                        if self.gpio_available and self.leds_initialized:
                            try:
                                GPIO.output(LED_RED, GPIO.HIGH)
                                self.red_led_state = True
                                print("   ✓ Red LED is now ON (ML detection)")
                                
                                # Start a timer to turn it off after 3 seconds
                                def delayed_off():
                                    time.sleep(3.0)
                                    if self.gpio_available and self.leds_initialized and not self.ml_attack_detected:
                                        GPIO.output(LED_RED, GPIO.LOW)
                                        self.red_led_state = False
                                        print("   ✓ Red LED turned OFF after 3 seconds (ML)")
                                
                                threading.Thread(target=delayed_off, daemon=True).start()
                                
                            except Exception as e:
                                print(f"   ✗ Failed to control red LED: {e}")
                        
            except Exception as e:
                print(f"Error in periodic IDS evaluation: {e}")

    def _ensure_complete_vehicle_data(self):
        """Ensure all required fields are populated with last known values"""
        complete_data = {}
        
        # Start with last known values
        for field, default_value in self.last_known_values.items():
            complete_data[field] = default_value
        
        # Update with current data (this overwrites defaults with actual current values)
        for field, value in self.current_vehicle_data.items():
            if value is not None:  # Only update if we have a real value
                complete_data[field] = value
                # Also update last known value
                self.last_known_values[field] = value
        
        return complete_data

    def _validate_data_for_ids(self, data):
        """Validate that data is suitable for IDS prediction"""
        # Check that all required fields are present
        required_fields = ["Speed (km/h)", "Battery Level (%)", "Throttle", "Brake", "Steering", "Gear"]
        for field in required_fields:
            if field not in data:
                return False
            
            # Check for valid numeric values (not None or NaN)
            value = data[field]
            if value is None:
                return False
            
            try:
                float_val = float(value)
                if np.isnan(float_val) or np.isinf(float_val):
                    return False
            except:
                return False
        
        return True

    def stop(self):
        """Stop TCU and cleanup GPIO"""
        print("\nStopping TCU...")
        self.running = False
        self.UDPrunning = False
        self.MQTTrunning = False
        
        # Turn off all LEDs
        if self.gpio_available and self.leds_initialized:
            GPIO.output(LED_RED, GPIO.LOW)
            GPIO.output(LED_GREEN, GPIO.LOW)
            GPIO.output(LED_BLUE, GPIO.LOW)
            GPIO.output(LED_YELLOW, GPIO.LOW)
        
        # Wait for threads
        if hasattr(self, 'listener_thread'):
            self.listener_thread.join(timeout=2)
        if hasattr(self, 'UDPlistener_thread'):
            self.UDPlistener_thread.join(timeout=2)
        if hasattr(self, 'ids_eval_thread'):
            self.ids_eval_thread.join(timeout=2)
        if hasattr(self, 'status_thread'):
            self.status_thread.join(timeout=2)
        
        # Stop MQTT and close CAN
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        if hasattr(self, 'can_bus'):
            self.can_bus.shutdown()
        
        # Cleanup GPIO
        if self.gpio_available and self.leds_initialized:
            GPIO.cleanup()
        
        print("TCU stopped.")

    # Include all the other methods from the original script
    def is_message_attacked(self, data):
        """Check if the message has the attack signature bit set"""
        if not data or len(data) == 0:
            return False
        return bool(data[-1] & self.SIGNATURE_BIT_MASK)

    def udp_listener(self, port=9876):
        """Listen for UDP packets"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        
        try:
            sock.bind(("0.0.0.0", port))
            print(f"Listening for UDP packets on port {port}...")
            
            while self.UDPrunning:
                try:
                    data, addr = sock.recvfrom(1024)
                    raw_message = data.decode().strip()
                    decoded_data = raw_message.split(",")
                    
                    if len(decoded_data) == 3:
                        x, y, z = map(float, decoded_data)
                        location_data = {"x": x, "y": y, "z": z}
                        self.last_location = location_data  # Store last location
                        print(f"Received Location Data: {location_data} from {addr}")
                        self._update_aggregated_data({"Location": location_data})
                    else:
                        print(f"Unexpected message format: {decoded_data} from {addr}")
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.UDPrunning:
                        print(f"UDP error: {e}")
                        
        finally:
            sock.close()
            print("UDP listener stopped.")

    def _decode_float(self, msg):
        """Decode float from CAN message"""
        try:
            return struct.unpack('>f', msg.data[:4])[0]
        except struct.error as e:
            print(f"Error decoding float: {e}")
            return None

    def _decode_int(self, msg):
        """Decode integer from CAN message"""
        try:
            return struct.unpack('>i', msg.data[:4])[0]
        except struct.error as e:
            print(f"Error decoding integer: {e}")
            return None

    def _round_floats(self, data):
        """Round floating-point values"""
        rounded_data = {}
        for key, value in data.items():
            if key == "Location":
                rounded_data[key] = {k: round(v, 2) if isinstance(v, float) else v 
                                   for k, v in value.items()}
            elif isinstance(value, dict):
                if "value" in value:
                    rounded_value = round(value["value"], 4) if isinstance(value["value"], float) else value["value"]
                    rounded_data[key] = {
                        "value": rounded_value,
                        "timestamp": value["timestamp"]
                    }
                    if "attacked" in value:
                        rounded_data[key]["attacked"] = value["attacked"]
                else:
                    rounded_data[key] = self._round_floats(value)
            elif isinstance(value, float):
                rounded_data[key] = round(value, 4)
            else:
                rounded_data[key] = value
        return rounded_data

    def _update_aggregated_data(self, new_data):
        """Update aggregated data and publish if changed"""
        # Check if ML attack is detected and add to data if needed
        if self.ids_enabled and self.ml_attack_detected:
            # Add ML attack info to the data
            if "ML_Attack_Detection" not in new_data:
                new_data["ML_Attack_Detection"] = {
                    "detected": True,
                    "confidence": round(self.ml_attack_confidence * 100, 2),
                    "method": "pattern_analysis"
                }
        
        if self._has_relevant_changes(new_data):
            self.last_sent_data.update(new_data)
            # Add last location if not in current update
            if "Location" not in self.last_sent_data:
                self.last_sent_data["Location"] = self.last_location
            self._publish_to_mqtt(self.last_sent_data)

    def _has_relevant_changes(self, new_data):
        """Check if data has relevant changes"""
        for key, value in new_data.items():
            if key not in self.last_sent_data:
                return True
                
            elif isinstance(value, dict):
                if not isinstance(self.last_sent_data[key], dict):
                    return True
                    
                if "value" in value:
                    if "value" not in self.last_sent_data[key]:
                        return True
                    if self.last_sent_data[key]["value"] != value["value"]:
                        return True
                        
                    if ("attacked" in value and "attacked" not in self.last_sent_data[key]) or \
                       ("attacked" in value and self.last_sent_data[key]["attacked"] != value["attacked"]):
                        return True
                else:
                    for sub_key, sub_value in value.items():
                        if sub_key not in self.last_sent_data[key] or self.last_sent_data[key][sub_key] != sub_value:
                            return True
                            
            elif self.last_sent_data[key] != value:
                return True
                
        return False

    def _publish_to_mqtt(self, data):
        """Publish data to MQTT and flash green LED"""
        topic = "vehicle/telemetry"
        
        # Create a copy for MQTT
        mqtt_data = {}
        for key, value in data.items():
            if isinstance(value, dict) and "raw_timestamp" in value:
                mqtt_data[key] = {k: v for k, v in value.items() if k != "raw_timestamp"}
            else:
                mqtt_data[key] = value
        
        rounded_data = self._round_floats(mqtt_data)
        payload = json.dumps(rounded_data)
        
        try:
            self.mqtt_client.publish(topic, payload)
            print(f"Published to MQTT: Topic={topic}, Payload={payload}")
            
            # Flash green LED for MQTT publish
            threading.Thread(target=self._flash_led, args=(LED_GREEN, 0.1)).start()
            
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")

    def _on_mqtt_connect(self, client, userdata, flags, reason_code, properties=None):
        """MQTT connect callback"""
        if hasattr(reason_code, 'is_failure'):
            if not reason_code.is_failure:
                print(f"Successfully connected to MQTT broker at {self.mqtt_broker_ip}")
                client.subscribe("vehicle/control")
            else:
                print(f"Failed to connect to MQTT broker: {reason_code}")
        else:
            if reason_code == 0:
                print(f"Successfully connected to MQTT broker at {self.mqtt_broker_ip}")
                client.subscribe("vehicle/control")
            else:
                print(f"Failed to connect to MQTT broker, return code {reason_code}")

    def _on_mqtt_disconnect(self, client, userdata, reason_code, properties=None):
        """MQTT disconnect callback"""
        if hasattr(reason_code, 'is_failure'):
            if reason_code.is_failure:
                print(f"Unexpected disconnection from MQTT broker: {reason_code}")
        else:
            if reason_code != 0:
                print(f"Unexpected disconnection from MQTT broker. Code: {reason_code}")

    def _on_mqtt_command(self, client, userdata, message):
        """Handle incoming MQTT commands"""
        try:
            payload = message.payload.decode()
            print(f"Received MQTT command: {payload}")
            if payload == "emergency_stop":
                print("Emergency Stop command received.")
                self.remote_brake = True
                self._send_can_command(0x130)
        except Exception as e:
            print(f"Error processing MQTT command: {e}")

    def _send_can_command(self, arbitration_id):
        """Send CAN command"""
        if self.remote_brake:
            try:
                msg = can.Message(arbitration_id=arbitration_id, data=[0x01], is_extended_id=False)
                self.can_bus.send(msg)
                print(f"Sent CAN Emergency Stop Command: ID={hex(arbitration_id)}")
            except can.CanError as e:
                print(f"Error sending CAN command: {e}")


# Helper functions remain the same
def setup_can_interface(interface='can0', bitrate=1000000):
    """Set up CAN interface on Raspberry Pi"""
    import subprocess
    
    print(f"Setting up {interface} with bitrate {bitrate}...")
    
    try:
        result = subprocess.run(['ip', 'link', 'show', interface], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: CAN interface {interface} not found")
            return False
            
        if 'UP' in result.stdout:
            print(f"{interface} is already UP")
            return True
    except Exception as e:
        print(f"Error checking interface: {e}")
    
    commands = [
        f"sudo ip link set {interface} down",
        f"sudo ip link set {interface} type can bitrate {bitrate}",
        f"sudo ip link set {interface} up"
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            print(f"✓ {cmd}")
            time.sleep(0.1)
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {cmd}")
            print(f"  Error: {e.stderr if e.stderr else e}")
            return False
    
    return True


def discover_mqtt_broker(network_range=None):
    """Discover MQTT broker on network"""
    import ipaddress
    import concurrent.futures
    
    def scan_port(ip, port=1883, timeout=0.5):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((str(ip), port))
            sock.close()
            return result == 0
        except:
            return False
    
    if network_range is None:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        network = ipaddress.ip_network(f"{local_ip}/24", strict=False)
    else:
        network = ipaddress.ip_network(network_range)
    
    print(f"Scanning {network} for MQTT brokers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        future_to_ip = {executor.submit(scan_port, str(ip)): str(ip) 
                       for ip in network.hosts()}
        
        for future in concurrent.futures.as_completed(future_to_ip):
            ip = future_to_ip[future]
            if future.result():
                print(f"Found MQTT broker at: {ip}")
                return ip
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TCU with IDS for Raspberry Pi')
    parser.add_argument('--can-interface', default='can0', 
                       help='CAN interface name (default: can0)')
    parser.add_argument('--mqtt-broker', default=None, 
                       help='MQTT broker IP address')
    parser.add_argument('--mqtt-port', type=int, default=1883, 
                       help='MQTT broker port (default: 1883)')
    parser.add_argument('--setup-can', action='store_true', 
                       help='Set up CAN interface before starting')
    parser.add_argument('--auto-discover', action='store_true',
                       help='Auto-discover MQTT broker on network')
    parser.add_argument('--test-ids', action='store_true',
                       help='Test IDS functionality before starting')
    parser.add_argument('--test-leds', action='store_true',
                       help='Test all LEDs before starting')
    parser.add_argument('--test-red-led', action='store_true',
                       help='Test just the red LED')
    
    args = parser.parse_args()
    
    # Test just red LED if requested
    if args.test_red_led:
        print("\n=== Testing Red LED Only ===")
        if GPIO_AVAILABLE:
            try:
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(LED_RED, GPIO.OUT)
                
                print(f"Red LED is on GPIO{LED_RED}")
                print("Turning on for 3 seconds...")
                GPIO.output(LED_RED, GPIO.HIGH)
                
                # Check if it's actually on
                state = GPIO.input(LED_RED)
                print(f"LED state after turning on: {state} (should be 1)")
                
                time.sleep(3)
                
                print("Turning off...")
                GPIO.output(LED_RED, GPIO.LOW)
                
                # Check if it's actually off
                state = GPIO.input(LED_RED)
                print(f"LED state after turning off: {state} (should be 0)")
                
                print("✓ Red LED test complete!")
                GPIO.cleanup()
                sys.exit(0)
            except Exception as e:
                print(f"✗ Red LED test failed: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        else:
            print("✗ GPIO not available on this system")
            sys.exit(1)
    
    # Test LEDs if requested
    if args.test_leds:
        print("\n=== Testing LEDs ===")
        if GPIO_AVAILABLE:
            try:
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(LED_RED, GPIO.OUT)
                GPIO.setup(LED_GREEN, GPIO.OUT)
                GPIO.setup(LED_BLUE, GPIO.OUT)
                GPIO.setup(LED_YELLOW, GPIO.OUT)
                
                leds = [
                    (LED_BLUE, "Blue (System running)"),
                    (LED_GREEN, "Green (MQTT upload)"),
                    (LED_YELLOW, "Yellow (IDS operational)"),
                    (LED_RED, "Red (Attack detected)")
                ]
                
                for led_pin, led_name in leds:
                    print(f"Testing {led_name}...")
                    GPIO.output(led_pin, GPIO.HIGH)
                    time.sleep(1)
                    GPIO.output(led_pin, GPIO.LOW)
                    time.sleep(0.5)
                
                print("✓ LED test complete!")
                GPIO.cleanup()
                sys.exit(0)
            except Exception as e:
                print(f"✗ LED test failed: {e}")
                sys.exit(1)
        else:
            print("✗ GPIO not available on this system")
            sys.exit(1)
    
    # Test IDS separately if requested
    if args.test_ids:
        print("\n=== Testing IDS Module ===")
        try:
            test_ids = VehicleIDS()
            
            # Test with normal data
            print("\nTesting with normal data...")
            normal_data = {
                "Speed (km/h)": 60.0,
                "Battery Level (%)": 85.0,
                "Throttle": 30.0,
                "Brake": 0.0,
                "Steering": 5.0,
                "Gear": 3
            }
            
            # Fill buffer
            for i in range(test_ids.window_size):
                is_attack, confidence = test_ids.predict(normal_data)
                print(f"  Prediction {i+1}: Attack={is_attack}, Confidence={confidence*100:.2f}%")
            
            # Test with attack data
            print("\nTesting with attack data (rapid changes)...")
            attack_data = normal_data.copy()
            for i in range(5):
                # Create rapid changes
                attack_data["Speed (km/h)"] += 60  # Rapid acceleration
                attack_data["Throttle"] = 100.0
                attack_data["Battery Level (%)"] -= 15  # Rapid battery drain
                
                is_attack, confidence = test_ids.predict(attack_data)
                print(f"  Attack test {i+1}: Attack={is_attack}, Confidence={confidence*100:.2f}%")
            
            print("\n✓ IDS module test completed!")
            sys.exit(0)
        except Exception as e:
            print(f"✗ IDS module test failed: {e}")
            sys.exit(1)
    
    # Auto-discover MQTT broker if needed
    if args.auto_discover or args.mqtt_broker is None:
        discovered_broker = discover_mqtt_broker()
        if discovered_broker:
            args.mqtt_broker = discovered_broker
            print(f"Using discovered MQTT broker: {args.mqtt_broker}")
        else:
            print("No MQTT broker found. Please specify with --mqtt-broker")
            sys.exit(1)
    
    # Set up CAN interface if requested
    if args.setup_can:
        if not setup_can_interface(args.can_interface):
            print("Failed to set up CAN interface. Exiting.")
            sys.exit(1)
    
    # Create and run TCU with IDS
    tcu = None
    try:
        tcu = TCU(can_interface=args.can_interface, 
                  mqtt_broker_ip=args.mqtt_broker, 
                  mqtt_port=args.mqtt_port)
        
        print("\nTCU with IDS is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C detected.")
    except Exception as e:
        print(f"\nUnhandled exception: {e}")
    finally:
        if tcu:
            tcu.stop()
        print("\nTCU shutdown complete.")