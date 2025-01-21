import glob
import os
import sys
import random
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# List to keep track of all actors for cleanup
actor_list = []

try:
    # Connect to the CARLA simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)

    # Get the world
    world = client.get_world()

    # Get the blueprint library and select a Tesla Model 3
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # Randomize the vehicle color if applicable
    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)

    # Choose a random spawn point
    transform = random.choice(world.get_map().get_spawn_points())

    # Spawn the vehicle
    vehicle = world.spawn_actor(bp, transform)
    vehicle.show_debug_telemetry(True)
    print(vehicle.get_speed_limit)
    actor_list.append(vehicle)
    print(f'Created vehicle: {vehicle.type_id}')

    # Enable autopilot for the vehicle
    vehicle.set_autopilot(True)

    # Start the timer
    start_time = time.time()
    duration = 300  # Run for 20 seconds

    print("Starting vehicle simulation...")

    while time.time() - start_time < duration:
        # Get vehicle data
        velocity = vehicle.get_velocity()
        speed = (3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5)  # Convert m/s to km/h
        control = vehicle.get_control()
        location = vehicle.get_location()

        # Print vehicle data
        print(f"Time: {time.time() - start_time:.2f} s")
        print(f"Speed: {speed:.2f} km/h")
        print(f"Control: Throttle={control.throttle}, Brake={control.brake}, Steering={control.steer}, gear={control.gear}")
        print(f"Location: {location}")
        print("-" * 40)
        

        # Wait for 1 second
        time.sleep(1)

finally:
    # Clean up all actors
    print("Destroying actors...")
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print("Done.")