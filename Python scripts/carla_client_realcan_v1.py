import carla
import struct
from datetime import datetime
import time
import random
import socket  # For UDP communication

# Import gs-usb modules for real CAN
from gs_usb.gs_usb import GsUsb
from gs_usb.gs_usb_frame import GsUsbFrame
from gs_usb.constants import (
    CAN_EFF_FLAG,
    CAN_ERR_FLAG,
    CAN_RTR_FLAG,
)

# gs_usb constants
GS_USB_ECHO_ID = 0
GS_USB_NONE_ECHO_ID = 0xFFFFFFFF

# CAN modes for gs_usb 0.3.0
GS_CAN_MODE_NORMAL = 0
GS_CAN_MODE_LISTEN_ONLY = (1 << 0)
GS_CAN_MODE_LOOP_BACK = (1 << 1)
GS_CAN_MODE_ONE_SHOT = (1 << 3)
GS_CAN_MODE_HW_TIMESTAMP = (1 << 4)

def connect_to_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)
    return client

def spawn_vehicle(client):
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)
    transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, transform)
    vehicle.set_autopilot(True)
    print(f"Spawned vehicle: {vehicle.type_id}")
    return vehicle

def update_battery(vehicle, battery_level):
    """
    Update the battery level based on throttle usage.
    Battery decreases proportionally to throttle input.
    """
    throttle = vehicle.get_control().throttle
    # Reduce battery level by throttle value per second
    battery_level -= throttle * 0.1
    return max(0, battery_level)  # Avoid negative battery

def get_vehicle_data(vehicle, battery_level):
    """
    Collect telemetry data from the vehicle.
    """
    velocity = vehicle.get_velocity()
    speed = 3.6 * ((velocity.x ** 2 + velocity.y ** 2 ) ** 0.5)
    control = vehicle.get_control()
    
    # Get vehicle location
    location = vehicle.get_location()
    
    # Update battery level
    battery_level = update_battery(vehicle, battery_level)
    
    return {
        "speed_kmh": round(speed, 2),
        "battery_level": round(battery_level, 2),
        "control": {
            "throttle": round(control.throttle, 2),
            "brake": round(control.brake, 2),
            "steering": round(control.steer, 2),
            "gear": control.gear,
        },
        "updated_battery_level": battery_level  # Return the updated battery level
    }

def setup_real_can(bitrate=100000):
    """
    Set up the real CAN bus using Innomaker USB2CAN device.
    """
    devs = GsUsb.scan()
    if len(devs) == 0:
        print("Can not find gs_usb device")
        return None
    
    # Use the first device (index 1), not the firmware upgrade interface
    dev = devs[1]
    print(f"Found USB2CAN device: {dev}")
    
    # Set bitrate
    if not dev.set_bitrate(bitrate):
        print("Can not set bitrate for gs_usb")
        return None
    
    # Start device in NORMAL mode for real CAN communication
    dev.start(GS_CAN_MODE_NORMAL)  # <-- THIS WAS THE ISSUE! Was LOOP_BACK
    print(f"Connected to real CAN network at {bitrate} bps")
    
    return dev

def send_can_data(can_device, vehicle, battery_level):
    """
    Send telemetry data over the real CAN bus using gs-usb.
    """
    data = get_vehicle_data(vehicle, battery_level)
    
    # Get current time for logging purposes only
    current_time = time.time()
    timestamp_str = datetime.fromtimestamp(current_time).strftime("%y,%m,%d-%H,%M,%S")
    print(f"Sending CAN data at: {timestamp_str}")
    
    # Define arbitration IDs for each attribute
    frames_data = [
        (0x123, struct.pack('>f', data["speed_kmh"])),          # Speed as float
        (0x124, struct.pack('>f', data["battery_level"])),       # Battery level as float
        (0x125, struct.pack('>f', data["control"]["throttle"])), # Throttle as float
        (0x126, struct.pack('>f', data["control"]["brake"])),    # Brake as float
        (0x127, struct.pack('>f', data["control"]["steering"])), # Steering as float
        (0x128, struct.pack('>i', data["control"]["gear"])),     # Gear as integer
    ]
    
    for can_id, packed_data in frames_data:
        # Ensure the most significant bit of the last byte is 0
        # This is to reserve it for attack detection
        last_byte_index = len(packed_data) - 1
        last_byte = packed_data[last_byte_index]
        modified_last_byte = last_byte & 0x7F
        
        # Create a new bytearray to modify the packed_data
        modified_packed_data = bytearray(packed_data)
        modified_packed_data[last_byte_index] = modified_last_byte
        
        try:
            # Create GsUsbFrame for sending
            frame = GsUsbFrame(can_id=can_id, data=bytes(modified_packed_data))
            
            if can_device.send(frame):
                # Print the original value for display purposes
                original_value = struct.unpack('>f', packed_data)[0] if can_id != 0x128 else struct.unpack('>i', packed_data)[0]
                print(f"Sent CAN Data: ID={hex(can_id)}, Value={original_value}")
            else:
                print(f"Failed to send CAN message with ID={hex(can_id)}")
                
        except Exception as e:
            print(f"Error sending CAN message: {e}")
    
    # Return the updated battery level
    return data["updated_battery_level"]

def setup_udp_socket():
    """
    Create a UDP socket for sending location data.
    """
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return udp_socket

def send_location_data(udp_socket, vehicle, target_ip="192.168.0.111", target_port=9876):
    """
    Send the vehicle's location data over UDP.
    """
    location = vehicle.get_location()
    location_data = f"{location.x},{location.y},{location.z}"
    
    # Send the location data over UDP
    udp_socket.sendto(location_data.encode(), (target_ip, target_port))
    print(f"Sent Location Data: {location_data} to {target_ip}:{target_port}")

def receive_feature_can(can_device, vehicle):
    """
    Listen for incoming CAN messages and apply vehicle controls if a message with ID 0x130 is received.
    """
    try:
        iframe = GsUsbFrame()
        # Non-blocking read with 1 second timeout
        if can_device.read(iframe, 1):
            # Filter out echo frames
            if iframe.echo_id == GS_USB_NONE_ECHO_ID:
                print(f"Received CAN Message: ID={hex(iframe.can_id)}, Data={iframe.data}")
                
                if iframe.can_id == 0x130:  # If message ID 0x130 is received, apply brakes
                    print("Applying emergency brakes!")
                    vehicle.set_autopilot(False)  # Disable autopilot
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    except Exception as e:
        print(f"CAN reception error: {e}")

def clean_up(vehicle):
    """
    Clean up CARLA actors.
    """
    print("Destroying actors...")
    vehicle.destroy()
    print("Actors destroyed.")

if __name__ == "__main__":
    # Connect to CARLA
    client = connect_to_carla()
    vehicle = spawn_vehicle(client)
    
    # Set up the real CAN bus using Innomaker USB2CAN
    can_device = setup_real_can(bitrate=100000)  # 1 Mbps
    if can_device is None:
        print("Failed to initialize CAN device")
        clean_up(vehicle)
        exit(1)
    
    # Set up the UDP socket
    udp_socket = setup_udp_socket()
    print("UDP socket created for sending location data.")
    
    battery_level = 100.0  # Initial battery percentage
    
    try:
        print("Starting CARLA and sending CAN messages...")
        while True:
            # Update and send CAN data
            battery_level = send_can_data(can_device, vehicle, battery_level)
            send_location_data(udp_socket, vehicle)  # Send location data via UDP
            receive_feature_can(can_device, vehicle)
            time.sleep(1)  # Simulate real-time updates
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Shutting down...")
    finally:
        can_device.stop()  # Stop the CAN device
        udp_socket.close()  # Close the UDP socket
        clean_up(vehicle)