import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import paho.mqtt.client as mqtt
from flask import Flask, jsonify
import threading
import time
import json
import os
import csv
from datetime import datetime

# Constants
LOG_FILE = "vehicle_data_log.json"
DATA_PACKET = {}  # Latest data packet
ACCUMULATED_DATA = []  # Accumulate data
ROLLING_WINDOW_SECONDS = 60
CSV_SAVE_INTERVAL = 600  # Shorter interval for testing (30 seconds)
FIRST_DATA_TIME = None
DATA_BUFFER = []
LOG_FOLDER = None
LAST_SAVE_TIME = None  # Track when last CSV was saved

def setup_logging_directory():
    logs_dir = "logs"
    try:
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Create a single folder per day
        date_folder = datetime.now().strftime('%Y-%m-%d')
        log_folder = os.path.join(logs_dir, date_folder)
        os.makedirs(log_folder, exist_ok=True)

        print(f"Logging directory: {log_folder}")
        return log_folder
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return None

def save_to_csv(data, start_time):
    """
    Save telemetry data to CSV with improved CAN timestamp handling.
    Now includes a single 'Attacked' column indicating if the message was attacked.
    """
    global LOG_FOLDER, LAST_SAVE_TIME
    
    if not data or not LOG_FOLDER:
        print("Warning: No data to save or no log folder")
        return
        
    try:
        # Format filename based on start time
        timestamp = datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S")
        csv_file = os.path.join(LOG_FOLDER, f"data_{timestamp}.csv")
        
        print(f"Saving CSV to {csv_file} with {len(data)} records")
        
        # Define CSV fields - now with a single Attacked column
        fieldnames = [
            "CANtime",      # Timestamp when CAN message was received by TCU
            "Speed (km/h)", 
            "Battery Level (%)",
            "Throttle", 
            "Brake", 
            "Steering", 
            "Gear",
            "Location_x", 
            "Location_y", 
            "Location_z",
            "Attacked"      # Single column indicating if the message was attacked
        ]
        
        # Create a backup of data for debugging if needed
        with open(os.path.join(LOG_FOLDER, f"debug_data_{timestamp}.json"), 'w') as f:
            json.dump(data, f, indent=2)
        
        # Write to CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for packet in data:
                # Initialize the row
                row = {}
                
                # Extract CAN timestamps for each attribute
                can_timestamps = {}
                
                # Default attack status to "no"
                row["Attacked"] = "no"
                
                # Find the CAN timestamp for each attribute based on new nested structure
                for key, value in packet.items():
                    if key in ["Speed (km/h)", "Battery Level (%)", "Throttle", "Brake", "Steering", "Gear"]:
                        if isinstance(value, dict) and "raw_timestamp" in value:
                            # Store the raw timestamp for this attribute
                            can_timestamps[key] = value["raw_timestamp"]
                        elif isinstance(value, dict) and "value" in value:
                            # If we have a value but no raw_timestamp, use the value
                            row[key] = value["value"]
                            
                            # Check if this field is attacked
                            if "attacked" in value and value["attacked"] == "yes":
                                row["Attacked"] = "yes"
                
                # Determine the CANtime to use (prioritize attribute with data)
                can_time = None
                
                # First try to get the timestamp from any attribute with data
                for key, timestamp in can_timestamps.items():
                    if key in packet and isinstance(packet[key], dict) and "value" in packet[key]:
                        if packet[key]["value"] is not None:
                            can_time = timestamp
                            break
                
                # If no valid timestamp found, use any timestamp available
                if can_time is None and can_timestamps:
                    can_time = next(iter(can_timestamps.values()))
                
                # Fallback: Use reception time if no CAN timestamp available
                if can_time is None:
                    can_time = packet.get("timestamp", time.time())
                
                # Store the determined CAN reception time
                row["CANtime"] = can_time
                
                # Add standard attribute values
                for key in ["Speed (km/h)", "Battery Level (%)", "Throttle", "Brake", "Steering", "Gear"]:
                    if key in packet:
                        if isinstance(packet[key], dict) and "value" in packet[key]:
                            # Get value from new nested structure
                            row[key] = packet[key]["value"]
                        elif isinstance(packet[key], (int, float, str)):
                            # Support for legacy format
                            row[key] = packet[key]
                
                # Add location data if available
                if "Location" in packet and isinstance(packet["Location"], dict):
                    loc = packet["Location"]
                    row.update({
                        "Location_x": loc.get("x"),
                        "Location_y": loc.get("y"),
                        "Location_z": loc.get("z")
                    })
                
                # Write the row to CSV
                writer.writerow(row)
                
        print(f"✅ CSV saved successfully: {csv_file} ({len(data)} records)")
        LAST_SAVE_TIME = time.time()
        
    except Exception as e:
        print(f"❌ CSV Error: {e}")

# Flask setup
server = Flask(__name__)
LOG_FOLDER = setup_logging_directory()

@server.route('/get_data')
def get_data():
    return jsonify(DATA_PACKET)

@server.route('/save_csv')
def manual_save_csv():
    """API endpoint to manually trigger CSV saving"""
    global DATA_BUFFER, FIRST_DATA_TIME
    
    if not DATA_BUFFER:
        return jsonify({"status": "error", "message": "No data to save"})
    
    save_time = FIRST_DATA_TIME if FIRST_DATA_TIME else time.time()
    save_to_csv(DATA_BUFFER, save_time)
    
    return jsonify({
        "status": "success", 
        "message": f"Saved {len(DATA_BUFFER)} records",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# MQTT setup
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

def on_mqtt_message(client, userdata, message):
    global DATA_PACKET, FIRST_DATA_TIME, DATA_BUFFER
    try:
        packet = json.loads(message.payload.decode())
        
        # Add dashboard reception timestamp
        timestamped_packet = {"timestamp": time.time(), **packet}
        DATA_PACKET = packet
        ACCUMULATED_DATA.append(timestamped_packet)
        DATA_BUFFER.append(timestamped_packet)
        
        # Debug output to show if any attacks are detected
        attacked_fields = []
        for key, value in packet.items():
            if isinstance(value, dict) and "attacked" in value and value["attacked"] == "yes":
                attacked_fields.append(key)
                
        if attacked_fields:
            print(f"⚠️ ATTACK DETECTED on: {', '.join(attacked_fields)} - Buffer size: {len(DATA_BUFFER)}")
        else:
            print(f"Received MQTT packet - Buffer size: {len(DATA_BUFFER)}")
        
        # Initialize first data time if not set
        if FIRST_DATA_TIME is None:
            FIRST_DATA_TIME = time.time()
            print(f"First data received at: {datetime.fromtimestamp(FIRST_DATA_TIME)}")
        
        # Check if it's time to save CSV
        current_time = time.time()
        elapsed_time = current_time - FIRST_DATA_TIME
        
        if FIRST_DATA_TIME and elapsed_time >= CSV_SAVE_INTERVAL:
            print(f"CSV save interval reached. Elapsed time: {elapsed_time:.2f}s. Saving {len(DATA_BUFFER)} records...")
            save_to_csv(DATA_BUFFER, FIRST_DATA_TIME)
            # Keep a copy of the last saved data for debugging
            with open(os.path.join(LOG_FOLDER, "last_save_data.json"), 'w') as f:
                json.dump(DATA_BUFFER, f, indent=2)
            DATA_BUFFER = []
            FIRST_DATA_TIME = None  # Reset for next interval
            
    except Exception as e:
        print(f"MQTT Error: {e}")

mqtt_client.on_message = on_mqtt_message
# mqtt_client.connect("localhost", 1883)
mqtt_client.connect("192.168.0.125", 1883)
mqtt_client.subscribe("vehicle/telemetry")
mqtt_client.loop_start()

# Dash setup
app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dashboard/')

app.layout = html.Div([
    html.H1("TCU Data Monitoring Dashboard"),
    
    # Status indicators
    html.Div([
        html.Div([
            html.H4("Dashboard Status"),
            html.Div(id="dashboard-status")
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'margin': '5px', 'flex': '1'}),
        
        html.Div([
            html.H4("Data Logging"),
            html.Div(id="logging-status"),
            html.Button("Save CSV Now", id="save-csv-btn", style={'margin-top': '10px'})
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'margin': '5px', 'flex': '1'}),
        
        # Add an attack status indicator
        html.Div([
            html.H4("Attack Status"),
            html.Div(id="attack-status")
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'margin': '5px', 'flex': '1'}),
    ], style={'display': 'flex', 'margin-bottom': '20px'}),
    
    # Main interface
    html.Label("Select Attribute:"),
    dcc.Dropdown(id='attribute-dropdown', clearable=False),
    dcc.Interval(id='interval-component', interval=2000),
    html.Div(id='dynamic-graph'),
    html.Button("Emergency Stop", id="emergency-btn", n_clicks=0, 
               style={'background-color': '#ff4444', 'color': 'white', 'margin-top': '15px'}),
    html.Div(id="status-msg"),
    
    # Add timestamp display section
    html.Div([
        html.H3("Timestamp Information"),
        html.Div(id="timestamp-info")
    ], style={'margin-top': '20px', 'padding': '10px', 'border': '1px solid #ddd'})
])

@app.callback(
    Output('dashboard-status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dashboard_status(n):
    return [
        html.P(f"Total data points: {len(ACCUMULATED_DATA)}"),
        html.P(f"Buffer size: {len(DATA_BUFFER)}"),
        html.P(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    ]

@app.callback(
    Output('logging-status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_logging_status(n):
    global FIRST_DATA_TIME, LAST_SAVE_TIME
    
    status = []
    
    if FIRST_DATA_TIME:
        elapsed = time.time() - FIRST_DATA_TIME
        next_save = CSV_SAVE_INTERVAL - elapsed
        
        status.append(html.P(f"First data: {datetime.fromtimestamp(FIRST_DATA_TIME).strftime('%H:%M:%S')}"))
        status.append(html.P(f"Next save in: {max(0, next_save):.1f}s"))
    else:
        status.append(html.P("Waiting for data..."))
    
    if LAST_SAVE_TIME:
        status.append(html.P(f"Last saved: {datetime.fromtimestamp(LAST_SAVE_TIME).strftime('%H:%M:%S')}"))
    
    return status

@app.callback(
    Output('attack-status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_attack_status(n):
    """Display status of any detected attacks"""
    if not DATA_PACKET:
        return html.P("No data received", style={'color': 'gray'})
    
    # Check for attack flags in the current data packet
    attacked_fields = []
    for key, value in DATA_PACKET.items():
        if isinstance(value, dict) and "attacked" in value and value["attacked"] == "yes":
            attacked_fields.append(key)
    
    if attacked_fields:
        status = [
            html.P("⚠️ ATTACK DETECTED!", style={'color': 'red', 'font-weight': 'bold'}),
            html.P(f"Affected fields: {', '.join(attacked_fields)}", style={'color': 'red'})
        ]
        return status
    else:
        return html.P("No attacks detected", style={'color': 'green'})

@app.callback(
    Output('attribute-dropdown', 'options'),
    Input('interval-component', 'n_intervals')
)
def update_dropdown(n):
    if DATA_PACKET:
        options = []
        for key in DATA_PACKET:
            if key == "Location":
                options.extend([
                    {'label': 'Location_x', 'value': 'Location_x'},
                    {'label': 'Location_y', 'value': 'Location_y'},
                    {'label': 'Location_z', 'value': 'Location_z'}
                ])
            elif key not in ["timestamp", "raw_timestamp"]:  # Skip internal timestamps
                options.append({'label': key, 'value': key})
        options.append({'label': 'Location (Grid)', 'value': 'Location_Grid'})
        return options
    return []

@app.callback(
    Output('dynamic-graph', 'children'),
    [Input('interval-component', 'n_intervals'), Input('attribute-dropdown', 'value')]
)
def update_graph(n, value):
    if not value:
        return html.Div("Select an attribute")
    
    if value == "Location_Grid":
        return dcc.Graph(figure=update_location_grid())
    
    return dcc.Graph(figure=update_line_chart(value))

@app.callback(
    Output('timestamp-info', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_timestamp_info(n):
    if not DATA_PACKET:
        return "No data available"
    
    info = []
    
    # Dashboard time
    dash_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    info.append(html.P(f"Dashboard Time: {dash_time}"))
    
    # Find a CAN timestamp in the nested structure
    can_timestamp = None
    can_timestamp_key = None
    attack_detected = False
    
    # Look for raw_timestamp in any attribute's nested structure
    for key, value in DATA_PACKET.items():
        if isinstance(value, dict):
            if "raw_timestamp" in value:
                can_timestamp = value["raw_timestamp"]
                can_timestamp_key = key
                
            # Check if this field is under attack
            if "attacked" in value and value["attacked"] == "yes":
                attack_detected = True
                info.append(html.P(f"⚠️ Attack detected on {key}!", style={'color': 'red', 'font-weight': 'bold'}))
            
            if can_timestamp and attack_detected:
                break
    
    # Display CAN time if found
    if can_timestamp is not None:
        if isinstance(can_timestamp, (int, float)):
            can_time = datetime.fromtimestamp(can_timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            info.append(html.P(f"CAN Time from {can_timestamp_key}: {can_time} (raw: {can_timestamp})", style={'color': 'blue'}))
        else:
            info.append(html.P(f"CAN Time from {can_timestamp_key}: {can_timestamp} (invalid format)", style={'color': 'red'}))
    
    # Calculate latency if we have both timestamps
    if ACCUMULATED_DATA and can_timestamp is not None and "timestamp" in ACCUMULATED_DATA[-1]:
        dash_timestamp = ACCUMULATED_DATA[-1]["timestamp"]
        
        # Make sure both are numbers
        if isinstance(can_timestamp, (int, float)) and isinstance(dash_timestamp, (int, float)):
            latency = dash_timestamp - can_timestamp
            info.append(html.P(f"CAN to Dashboard Latency: {latency:.6f} seconds", 
                            style={'color': 'red' if latency > 1 else 'green'}))
    
    return info

def update_line_chart(attribute):
    if not ACCUMULATED_DATA:
        return {}
        
    values = []
    colors = []  # Colors to indicate attacked points
    
    for packet in ACCUMULATED_DATA:
        try:
            is_attacked = False
            value = None
            
            if attribute.startswith("Location_"):
                axis = attribute.split("_")[1]
                if "Location" in packet and isinstance(packet["Location"], dict):
                    value = packet["Location"].get(axis, 0)
            else:
                if attribute in packet:
                    if isinstance(packet[attribute], dict) and "value" in packet[attribute]:
                        # Handle nested structure
                        value = packet[attribute]["value"]
                        
                        # Check if this point is under attack
                        if "attacked" in packet[attribute] and packet[attribute]["attacked"] == "yes":
                            is_attacked = True
                    else:
                        # Handle flat structure (backwards compatibility)
                        value = packet[attribute]
                
            # Default value if attribute not found
            if value is None:
                value = 0
                
            values.append(value)
            colors.append('red' if is_attacked else 'blue')
                
        except Exception as e:
            print(f"Error processing data for line chart: {e}")
            values.append(0)  # Default value on error
            colors.append('blue')
    
    # Create scatter plot instead of line to support individual point colors
    return {
        "data": [{
            "x": list(range(len(values))), 
            "y": values, 
            "type": "scatter",
            "mode": "lines+markers",
            "marker": {"color": colors},
            "line": {"color": "lightblue"}
        }],
        "layout": {
            "title": f"{attribute} (Red points indicate attack)",
            "xaxis": {"title": "Time"}, 
            "yaxis": {"title": attribute}
        }
    }

def update_location_grid():
    """
    Simplified location grid function that displays vehicle path without time filtering.
    This version bypasses timestamp checks and displays all available location data.
    """
    global ACCUMULATED_DATA
    
    if not ACCUMULATED_DATA:
        return {
            "data": [],
            "layout": {
                "title": "Vehicle Location (No Data)",
                "xaxis": {"range": [-114.60, 109.98]},
                "yaxis": {"range": [-68.73, 141.21]}
            }
        }
    
    # Extract x,y coordinates from all data points
    x_coords = []
    y_coords = []
    
    for packet in ACCUMULATED_DATA:
        try:
            if "Location" in packet and isinstance(packet["Location"], dict):
                location = packet["Location"]
                
                x = location.get("x")
                y = location.get("y")
                
                if x is not None and y is not None:
                    x_coords.append(x)
                    y_coords.append(y)
        except Exception as e:
            print(f"Error extracting location: {e}")
    
    if not x_coords or not y_coords:
        return {
            "data": [],
            "layout": {
                "title": "Vehicle Location (No Coordinates Found)",
                "xaxis": {"range": [-114.60, 109.98]},
                "yaxis": {"range": [-68.73, 141.21]}
            }
        }
    
    # Create the path trace (dotted blue line)
    path_trace = {
        "x": x_coords,
        "y": y_coords,
        "mode": "lines",
        "name": "Path",
        "line": {
            "color": "blue",
            "dash": "dash",
            "width": 2
        }
    }
    
    # Create the current position trace (red dot)
    current_position_trace = {
        "x": [x_coords[-1]],
        "y": [y_coords[-1]],
        "mode": "markers",
        "name": "Current Position",
        "marker": {
            "color": "red",
            "size": 12,
            "symbol": "circle"
        }
    }
    
    return {
        "data": [path_trace, current_position_trace],
        "layout": {
            "title": f"Vehicle Location (Showing {len(x_coords)} points)",
            "xaxis": {"title": "X Position", "range": [-114.60, 109.98]},
            "yaxis": {"title": "Y Position", "range": [-68.73, 141.21]},
            "legend": {"x": 0, "y": 1.1, "orientation": "h"},
            "showlegend": True,
            "hovermode": "closest"
        }
    }

@app.callback(
    Output("status-msg", "children"),
    Input("emergency-btn", "n_clicks")
)
def emergency_stop(n):
    if n > 0:
        mqtt_client.publish("vehicle/control", "emergency_stop")
        return "Emergency stop sent!"
    return ""

@app.callback(
    Output("logging-status", "children", allow_duplicate=True),
    Input("save-csv-btn", "n_clicks"),
    prevent_initial_call=True
)
def save_csv_now(n_clicks):
    if n_clicks:
        if DATA_BUFFER:
            save_to_csv(DATA_BUFFER, time.time())
            return [
                html.P("CSV saved manually!", style={"color": "green", "font-weight": "bold"}),
                html.P(f"Saved {len(DATA_BUFFER)} records")
            ]
        else:
            return [
                html.P("No data to save!", style={"color": "red"})
            ]
    return []

if __name__ == "__main__":
    try:
        app.run_server(debug=True)
    except Exception as e:
        print(f"Error: {e}")