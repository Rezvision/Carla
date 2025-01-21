from flask import Flask, jsonify, request
import threading
import time
import json

# Constants
LOG_FILE = "vehicle_data_log.json"
DATA_PACKET = {}  # For the latest data packet
ACCUMULATED_DATA = []  # For accumulating data over time

# Flask setup
app = Flask(__name__)

# Route for the root URL (Optional, to prevent 404 errors)
@app.route('/')
def index():
    return "Flask Server is Running"

# Route for favicon (Optional, to prevent 404 errors)
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content for the favicon

# Function to start the Flask server
def start_flask():
    # Route to serve the most recent CAN data packet
    @app.route('/get_data', methods=['GET'])
    def get_data():
        return jsonify(DATA_PACKET), 200

    # Route to update the data packet
    @app.route('/update_data', methods=['POST'])
    def update_data():
        global DATA_PACKET
        packet = request.get_json()  # Assuming you're sending JSON data
        if packet:
            DATA_PACKET = packet  # Update global data packet
            ACCUMULATED_DATA.append(packet)  # Accumulate the data
            return jsonify({'status': 'success', 'message': 'Data updated successfully'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    print("Flask server started...")
    app.run(debug=True, use_reloader=False)  # Disable reloader to avoid conflicts with threading

# Function to log data to a JSON file
def log_data():
    try:
        with open(LOG_FILE, "a") as file:
            for packet in ACCUMULATED_DATA:
                json.dump(packet, file)
                file.write("\n")  # Add newline for each packet
        # Clear the accumulated data after saving to file
        ACCUMULATED_DATA.clear()
    except Exception as e:
        print(f"Error logging data: {e}")

# Function to periodically save the accumulated data every minute
def save_data_periodically():
    while True:
        time.sleep(60)  # Sleep for 60 seconds (1 minute)
        log_data()  # Save the accumulated data to the log file

# Main entry point to start Flask in a separate thread
if __name__ == "__main__":
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.start()

    # Start the periodic data saving function in a separate thread
    data_saving_thread = threading.Thread(target=save_data_periodically)
    data_saving_thread.start()

    # Keep the main thread alive while the Flask thread is running
    while True:
        time.sleep(1)  # Keep the main thread alive
