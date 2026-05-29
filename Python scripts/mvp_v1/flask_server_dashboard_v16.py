# flask_server_dashboard_v16.py
#
# Changes from v15:
#   • Emergency-stop / resume controls removed entirely (no MQTT publishes
#     on vehicle/<id>/control, no buttons, no callback, no status-msg div).
#   • Anomaly panel simplified to a single line per vehicle, of the form
#       "Anomaly detected in vehicle id: edge_1"
#     listing every vehicle currently flagged. No score/threshold display.
#   • Top-level vehicle dropdown retained as in v15.
#   • Everything else (per-vehicle state, CSV save logic, charts, MQTT
#     wildcard subscription, score-vs-threshold chart) unchanged.

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
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────
VEHICLES = ["edge_1", "edge_2", "edge_3"]   # must match fed_client --client-id

MQTT_BROKER = "127.0.0.1"
MQTT_PORT   = 1883
TOPIC_TELEM_WILDCARD = "vehicle/+/telemetry"

ROLLING_WINDOW_SECONDS = 60
CSV_SAVE_INTERVAL      = 600       # auto-save every 10 minutes per vehicle

# ── Per-vehicle state (defaultdict so unknown ids don't crash) ───────────────
DATA_PACKET      = defaultdict(dict)             # latest packet per vehicle
ACCUMULATED_DATA = defaultdict(list)             # full history per vehicle
DATA_BUFFER      = defaultdict(list)             # buffer flushed to CSV
FIRST_DATA_TIME  = defaultdict(lambda: None)     # first-packet ts per vehicle
LAST_SAVE_TIME   = defaultdict(lambda: None)     # last successful CSV save

LOG_FOLDER = None


# ── Logging directory setup ──────────────────────────────────────────────────
def setup_logging_directory():
    logs_dir = "logs"
    try:
        os.makedirs(logs_dir, exist_ok=True)
        date_folder = datetime.now().strftime('%Y-%m-%d')
        log_folder  = os.path.join(logs_dir, date_folder)
        os.makedirs(log_folder, exist_ok=True)
        for v in VEHICLES:
            os.makedirs(os.path.join(log_folder, v), exist_ok=True)
        print(f"Logging directory: {log_folder}")
        return log_folder
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return None


# ── CSV save (per vehicle) ───────────────────────────────────────────────────
def save_to_csv(client_id, data, start_time):
    """Save telemetry for one vehicle. Anomaly flag is window-level."""
    global LOG_FOLDER

    if not data or not LOG_FOLDER:
        print(f"[{client_id}] Warning: no data to save or no log folder")
        return

    try:
        timestamp      = datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S")
        vehicle_folder = os.path.join(LOG_FOLDER, client_id)
        os.makedirs(vehicle_folder, exist_ok=True)
        csv_file       = os.path.join(vehicle_folder, f"data_{timestamp}.csv")

        print(f"[{client_id}] Saving CSV to {csv_file} ({len(data)} records)")

        fieldnames = [
            "Timestamp",
            "Speed (km/h)",
            "Battery Level (%)",
            "Throttle",
            "Brake",
            "Steering",
            "Gear",
            "Location_x",
            "Location_y",
            "Location_z",
            "Anomaly",
            "AnomalyScore",
            "Threshold",
        ]

        with open(os.path.join(vehicle_folder, f"debug_data_{timestamp}.json"), 'w') as f:
            json.dump(data, f, indent=2)

        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for packet in data:
                row = {}
                row["Timestamp"] = packet.get("timestamp", time.time())

                values = packet.get("values", {}) or {}
                for key in ["Speed (km/h)", "Battery Level (%)", "Throttle",
                            "Brake", "Steering", "Gear"]:
                    row[key] = values.get(key)

                loc = packet.get("Location", {}) or {}
                row["Location_x"] = loc.get("x")
                row["Location_y"] = loc.get("y")
                row["Location_z"] = loc.get("z")

                anomaly = packet.get("anomaly", {}) or {}
                row["Anomaly"]      = "yes" if anomaly.get("detected") else "no"
                row["AnomalyScore"] = anomaly.get("score")
                row["Threshold"]    = anomaly.get("threshold")

                writer.writerow(row)

        print(f"✅ [{client_id}] CSV saved: {csv_file}")
        LAST_SAVE_TIME[client_id] = time.time()

    except Exception as e:
        print(f"❌ [{client_id}] CSV error: {e}")


# ── Flask ────────────────────────────────────────────────────────────────────
server     = Flask(__name__)
LOG_FOLDER = setup_logging_directory()


@server.route('/get_data')
def get_data_all():
    return jsonify({v: DATA_PACKET[v] for v in VEHICLES})


@server.route('/get_data/<client_id>')
def get_data_vehicle(client_id):
    return jsonify(DATA_PACKET.get(client_id, {}))


@server.route('/save_csv/<client_id>')
def manual_save_csv(client_id):
    if client_id not in VEHICLES:
        return jsonify({"status": "error", "message": "unknown vehicle"})
    if not DATA_BUFFER[client_id]:
        return jsonify({"status": "error", "message": "no data to save"})
    save_time = FIRST_DATA_TIME[client_id] or time.time()
    save_to_csv(client_id, DATA_BUFFER[client_id], save_time)
    return jsonify({
        "status":  "success",
        "message": f"Saved {len(DATA_BUFFER[client_id])} records for {client_id}",
        "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# ── MQTT ─────────────────────────────────────────────────────────────────────
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)


def _extract_client_id(topic, packet):
    """vehicle/<client_id>/telemetry → client_id, with payload fallback."""
    parts = topic.split("/")
    if len(parts) >= 3 and parts[0] == "vehicle":
        return parts[1]
    return packet.get("client_id", "unknown")


def on_mqtt_message(client, userdata, message):
    try:
        packet    = json.loads(message.payload.decode())
        client_id = _extract_client_id(message.topic, packet)

        if client_id not in VEHICLES:
            print(f"[MQTT] Unknown vehicle '{client_id}' on {message.topic} — accepting")

        timestamped = {"timestamp": time.time(), **packet}
        DATA_PACKET[client_id] = packet
        ACCUMULATED_DATA[client_id].append(timestamped)
        DATA_BUFFER[client_id].append(timestamped)

        anomaly = packet.get("anomaly", {}) or {}
        if anomaly.get("detected"):
            score = anomaly.get("score", 0.0)
            thr   = anomaly.get("threshold", 0.0)
            print(f"⚠️  [{client_id}] ANOMALY  score={score:.6f}  thr={thr:.6f}  "
                  f"buffer={len(DATA_BUFFER[client_id])}")
        else:
            if len(ACCUMULATED_DATA[client_id]) % 50 == 0:
                print(f"[{client_id}] healthy  buffer={len(DATA_BUFFER[client_id])}")

        if FIRST_DATA_TIME[client_id] is None:
            FIRST_DATA_TIME[client_id] = time.time()
            print(f"[{client_id}] first data at "
                  f"{datetime.fromtimestamp(FIRST_DATA_TIME[client_id])}")

        elapsed = time.time() - FIRST_DATA_TIME[client_id]
        if elapsed >= CSV_SAVE_INTERVAL:
            print(f"[{client_id}] CSV interval reached — saving "
                  f"{len(DATA_BUFFER[client_id])} records")
            save_to_csv(client_id, DATA_BUFFER[client_id],
                        FIRST_DATA_TIME[client_id])
            DATA_BUFFER[client_id]     = []
            FIRST_DATA_TIME[client_id] = None

    except Exception as e:
        print(f"[MQTT] message error: {e}")


mqtt_client.on_message = on_mqtt_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.subscribe(TOPIC_TELEM_WILDCARD)
mqtt_client.loop_start()
print(f"[MQTT] subscribed to {TOPIC_TELEM_WILDCARD} on {MQTT_BROKER}:{MQTT_PORT}")


# ── Dash setup ───────────────────────────────────────────────────────────────
app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dashboard/')

app.layout = html.Div([
    html.H1("Federated IDS — TCU Monitoring Dashboard (v16)"),

    # ── Top-level vehicle selector ───────────────────────────────────────────
    html.Div([
        html.Label("Vehicle:", style={'font-weight': 'bold',
                                      'margin-right': '10px',
                                      'font-size': '16px'}),
        dcc.Dropdown(
            id='vehicle-dropdown',
            options=[{'label': v, 'value': v} for v in VEHICLES],
            value=VEHICLES[0],
            clearable=False,
            style={'width': '300px', 'display': 'inline-block'}
        ),
    ], style={'padding': '12px', 'border': '2px solid #333',
              'margin-bottom': '20px', 'background-color': '#f0f0f0'}),

    # ── Global anomaly banner (across all vehicles) ──────────────────────────
    html.Div(id="global-anomaly-banner",
             style={'padding': '12px', 'margin-bottom': '20px',
                    'border': '1px solid #ddd', 'font-size': '16px'}),

    # ── Status row ───────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H4("Dashboard Status"),
            html.Div(id="dashboard-status")
        ], style={'padding': '10px', 'border': '1px solid #ddd',
                  'margin': '5px', 'flex': '1'}),

        html.Div([
            html.H4("Data Logging"),
            html.Div(id="logging-status"),
            html.Button("Save CSV Now", id="save-csv-btn",
                        style={'margin-top': '10px'})
        ], style={'padding': '10px', 'border': '1px solid #ddd',
                  'margin': '5px', 'flex': '1'}),
    ], style={'display': 'flex', 'margin-bottom': '20px'}),

    # ── Attribute selector + live chart ──────────────────────────────────────
    html.Label("Select Attribute:"),
    dcc.Dropdown(id='attribute-dropdown', clearable=False),
    dcc.Interval(id='interval-component', interval=2000),
    html.Div(id='dynamic-graph'),

    # ── Anomaly score chart ──────────────────────────────────────────────────
    html.Div([
        html.H3("Anomaly Score (reconstruction error vs threshold)"),
        dcc.Graph(id='anomaly-score-chart')
    ], style={'margin-top': '20px', 'padding': '10px',
              'border': '1px solid #ddd'}),

    # ── Timestamp panel ──────────────────────────────────────────────────────
    html.Div([
        html.H3("Timestamp Information"),
        html.Div(id="timestamp-info")
    ], style={'margin-top': '20px', 'padding': '10px',
              'border': '1px solid #ddd'})
])


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output('global-anomaly-banner', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_global_anomaly_banner(n):
    """Single-line summary: which vehicles (if any) are currently flagged."""
    flagged = []
    for v in VEHICLES:
        packet  = DATA_PACKET[v]
        anomaly = packet.get("anomaly", {}) or {}
        if anomaly.get("detected"):
            flagged.append(v)

    if not flagged:
        return html.Span("No anomalies detected.",
                         style={'color': 'green', 'font-weight': 'bold'})

    # One line per flagged vehicle, in the exact phrasing asked for
    return [
        html.Div(f"⚠️ Anomaly detected in vehicle id: {v}",
                 style={'color': 'red', 'font-weight': 'bold'})
        for v in flagged
    ]


@app.callback(
    Output('dashboard-status', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('vehicle-dropdown', 'value')]
)
def update_dashboard_status(n, vehicle):
    if not vehicle:
        return html.P("No vehicle selected")
    return [
        html.P(f"Vehicle: {vehicle}", style={'font-weight': 'bold'}),
        html.P(f"Total data points: {len(ACCUMULATED_DATA[vehicle])}"),
        html.P(f"Buffer size: {len(DATA_BUFFER[vehicle])}"),
        html.P(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    ]


@app.callback(
    Output('logging-status', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('vehicle-dropdown', 'value')]
)
def update_logging_status(n, vehicle):
    if not vehicle:
        return html.P("No vehicle selected")

    status = []
    first  = FIRST_DATA_TIME[vehicle]
    last   = LAST_SAVE_TIME[vehicle]

    if first:
        elapsed   = time.time() - first
        next_save = CSV_SAVE_INTERVAL - elapsed
        status.append(html.P(f"First data: "
                             f"{datetime.fromtimestamp(first).strftime('%H:%M:%S')}"))
        status.append(html.P(f"Next save in: {max(0, next_save):.1f}s"))
    else:
        status.append(html.P("Waiting for data..."))

    if last:
        status.append(html.P(f"Last saved: "
                             f"{datetime.fromtimestamp(last).strftime('%H:%M:%S')}"))
    return status


@app.callback(
    Output('attribute-dropdown', 'options'),
    [Input('interval-component', 'n_intervals'),
     Input('vehicle-dropdown', 'value')]
)
def update_dropdown(n, vehicle):
    if not vehicle or not DATA_PACKET[vehicle]:
        return []
    packet  = DATA_PACKET[vehicle]
    values  = packet.get("values", {}) or {}
    options = [{'label': k, 'value': k} for k in values.keys()]
    if "Location" in packet:
        options.extend([
            {'label': 'Location_x',      'value': 'Location_x'},
            {'label': 'Location_y',      'value': 'Location_y'},
            {'label': 'Location_z',      'value': 'Location_z'},
            {'label': 'Location (Grid)', 'value': 'Location_Grid'},
        ])
    return options


@app.callback(
    Output('dynamic-graph', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('attribute-dropdown', 'value'),
     Input('vehicle-dropdown', 'value')]
)
def update_graph(n, attribute, vehicle):
    if not attribute or not vehicle:
        return html.Div("Select a vehicle and attribute")
    if attribute == "Location_Grid":
        return dcc.Graph(figure=update_location_grid(vehicle))
    return dcc.Graph(figure=update_line_chart(vehicle, attribute))


@app.callback(
    Output('anomaly-score-chart', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('vehicle-dropdown', 'value')]
)
def update_anomaly_chart(n, vehicle):
    if not vehicle or not ACCUMULATED_DATA[vehicle]:
        return {"data": [], "layout": {"title": "No anomaly data yet"}}

    scores, thresholds, colors = [], [], []
    for packet in ACCUMULATED_DATA[vehicle][-300:]:
        a = packet.get("anomaly", {}) or {}
        s = a.get("score")
        t = a.get("threshold")
        if isinstance(s, (int, float)):
            scores.append(s)
            thresholds.append(t if isinstance(t, (int, float)) else None)
            colors.append('red' if a.get("detected") else 'blue')

    if not scores:
        return {"data": [], "layout": {"title": "No anomaly scores yet"}}

    x = list(range(len(scores)))
    return {
        "data": [
            {
                "x": x, "y": scores,
                "type": "scatter", "mode": "lines+markers",
                "name": "Reconstruction error",
                "marker": {"color": colors},
                "line":   {"color": "lightblue"}
            },
            {
                "x": x, "y": thresholds,
                "type": "scatter", "mode": "lines",
                "name": "Threshold",
                "line": {"color": "orange", "dash": "dash"}
            }
        ],
        "layout": {
            "title":  f"{vehicle}: anomaly score (red = anomaly detected)",
            "xaxis":  {"title": "Window index"},
            "yaxis":  {"title": "Reconstruction error"},
        }
    }


@app.callback(
    Output('timestamp-info', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('vehicle-dropdown', 'value')]
)
def update_timestamp_info(n, vehicle):
    if not vehicle or not DATA_PACKET[vehicle]:
        return "No data available"

    info      = []
    dash_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    info.append(html.P(f"Dashboard Time: {dash_time}"))

    packet = DATA_PACKET[vehicle]
    src_ts = packet.get("timestamp")
    if isinstance(src_ts, (int, float)):
        src_time = datetime.fromtimestamp(src_ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        info.append(html.P(f"Source Time ({vehicle}): {src_time} "
                           f"(raw: {src_ts})", style={'color': 'blue'}))

        if ACCUMULATED_DATA[vehicle]:
            recv_ts = ACCUMULATED_DATA[vehicle][-1].get("timestamp")
            if isinstance(recv_ts, (int, float)):
                latency = recv_ts - src_ts
                info.append(html.P(
                    f"Edge → Dashboard latency: {latency:.6f} s",
                    style={'color': 'red' if latency > 1 else 'green'}))
    return info


# ── Chart helpers ────────────────────────────────────────────────────────────

def update_line_chart(vehicle, attribute):
    """Selected attribute over time for one vehicle.
       Red markers = the window the IDS flagged anomalous."""
    if not ACCUMULATED_DATA[vehicle]:
        return {}

    values, colors = [], []
    for packet in ACCUMULATED_DATA[vehicle]:
        try:
            anomaly = packet.get("anomaly", {}) or {}
            is_anom = bool(anomaly.get("detected"))

            value = None
            if attribute.startswith("Location_"):
                axis = attribute.split("_")[1]
                loc  = packet.get("Location", {}) or {}
                value = loc.get(axis, 0)
            else:
                vals  = packet.get("values", {}) or {}
                value = vals.get(attribute, 0)

            if value is None:
                value = 0
            values.append(value)
            colors.append('red' if is_anom else 'blue')
        except Exception as e:
            print(f"[{vehicle}] line chart error: {e}")
            values.append(0)
            colors.append('blue')

    return {
        "data": [{
            "x":    list(range(len(values))),
            "y":    values,
            "type": "scatter",
            "mode": "lines+markers",
            "marker": {"color": colors},
            "line":   {"color": "lightblue"}
        }],
        "layout": {
            "title": f"{vehicle} — {attribute} (red = anomalous window)",
            "xaxis": {"title": "Time"},
            "yaxis": {"title": attribute}
        }
    }


def update_location_grid(vehicle):
    """Vehicle path. Anomalous windows marked with red ✕."""
    if not ACCUMULATED_DATA[vehicle]:
        return {
            "data":   [],
            "layout": {
                "title": f"{vehicle} location (no data)",
                "xaxis": {"range": [-114.60, 109.98]},
                "yaxis": {"range": [-68.73, 141.21]}
            }
        }

    x_coords, y_coords, anom_x, anom_y = [], [], [], []
    for packet in ACCUMULATED_DATA[vehicle]:
        loc  = packet.get("Location", {}) or {}
        x, y = loc.get("x"), loc.get("y")
        if x is None or y is None:
            continue
        x_coords.append(x)
        y_coords.append(y)
        anomaly = packet.get("anomaly", {}) or {}
        if anomaly.get("detected"):
            anom_x.append(x)
            anom_y.append(y)

    if not x_coords:
        return {"data": [],
                "layout": {"title": f"{vehicle} location (no coordinates)"}}

    traces = [
        {
            "x": x_coords, "y": y_coords,
            "mode": "lines", "name": "Path",
            "line": {"color": "blue", "dash": "dash", "width": 2}
        },
        {
            "x": anom_x, "y": anom_y,
            "mode": "markers", "name": "Anomalies",
            "marker": {"color": "red", "size": 8, "symbol": "x"}
        },
        {
            "x": [x_coords[-1]], "y": [y_coords[-1]],
            "mode": "markers", "name": "Current",
            "marker": {"color": "green", "size": 12, "symbol": "circle"}
        }
    ]

    return {
        "data": traces,
        "layout": {
            "title":  f"{vehicle} location ({len(x_coords)} points, "
                      f"{len(anom_x)} anomalies)",
            "xaxis":  {"title": "X Position", "range": [-114.60, 109.98]},
            "yaxis":  {"title": "Y Position", "range": [-68.73, 141.21]},
            "legend": {"x": 0, "y": 1.1, "orientation": "h"},
            "showlegend": True,
            "hovermode":  "closest"
        }
    }


# ── Manual CSV save button ───────────────────────────────────────────────────

@app.callback(
    Output('logging-status', 'children', allow_duplicate=True),
    Input('save-csv-btn', 'n_clicks'),
    State('vehicle-dropdown', 'value'),
    prevent_initial_call=True
)
def save_csv_now(n_clicks, vehicle):
    if not n_clicks:
        return []
    if not vehicle:
        return [html.P("No vehicle selected!", style={"color": "red"})]
    if DATA_BUFFER[vehicle]:
        save_to_csv(vehicle, DATA_BUFFER[vehicle], time.time())
        return [
            html.P(f"CSV saved manually for {vehicle}!",
                   style={"color": "green", "font-weight": "bold"}),
            html.P(f"Saved {len(DATA_BUFFER[vehicle])} records")
        ]
    return [html.P(f"No data to save for {vehicle}!", style={"color": "red"})]


if __name__ == "__main__":
    try:
        app.run(debug=True, host="127.0.0.1")
    except Exception as e:
        print(f"Error: {e}")
