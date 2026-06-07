from collections import defaultdict
import os
import cv2
import time
import sqlite3
import threading
from flask import (Flask, render_template, request, redirect, url_for,
                session, flash, Response, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import numpy as np
import uuid

# ── Optional YOLOv8 import ─────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    model = YOLO("yolov8n.pt")
    model.lock = threading.Lock() 
    print(" [SUCCESS] YOLOv8 model loaded successfully with dedicated instance lock.")
except Exception as e:
    YOLO_AVAILABLE = False
    print(f" [ERROR] YOLO model loading failed: {e}")
    model = None

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'videos')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATABASE = os.path.join(os.path.dirname(__file__), "database.db")
VIDEO_DIR = os.path.join(app.root_path, "static", "videos")

ROADS = {
    "indore":   {"label": "Indore Rd",       "color": (0, 200, 255)},
    "dewas":    {"label": "Dewas Rd",        "color": (0, 255, 128)},
    "ramghat":  {"label": "Ram Ghat Rd",     "color": (255, 128, 0)},
    "dutt":     {"label": "Dutt Akhand Rd",  "color": (200, 0, 255)},
}

PARKING = {
    "lot_1": {"label": "Zone A - North", "total": 500, "occupancy": 320, "inflow": 0, "outflow": 0, "predicted_occupancy": 320},
    "lot_2": {"label": "Zone B - South", "total": 300, "occupancy": 150, "inflow": 0, "outflow": 0, "predicted_occupancy": 150},
    "lot_3": {"label": "Zone C - East",  "total": 450, "occupancy": 100, "inflow": 0, "outflow": 0, "predicted_occupancy": 100},
    "lot_4": {"label": "Zone D - West",  "total": 600, "occupancy": 550, "inflow": 0, "outflow": 0, "predicted_occupancy": 550},
}

COUNTING_LINE_Y_RATIO = 0.5   # fraction of frame height


# ── Per-stream state ───────────────────────────────────────────────────────
stream_state: dict = {}
stream_lock = threading.Lock()

PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

stream_state = {}
stream_lock = threading.Lock()

def get_or_init_state(stream_id: str) -> dict:
    with stream_lock:
        if stream_id not in stream_state:
            stream_state[stream_id] = {
                "vehicle_count": 0,
                "person_count": 0,
                "counted_ids": set(),
                "track_history": defaultdict(list)
            }
        return stream_state[stream_id]

# ── Database ───────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                email    TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL
            );
            
            -- ADD THIS LINE TEMPORARILY:
            DROP TABLE IF EXISTS bookings; 
            
            CREATE TABLE bookings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                pickup      TEXT    NOT NULL,
                seats       INTEGER NOT NULL,
                travel_date TEXT    NOT NULL,
                travel_time TEXT    NOT NULL,
                taxi_number TEXT    NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
init_db()
# ── Auth helpers ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── Frame generation helpers ───────────────────────────────────────────────
def _draw_overlay(frame, state: dict, label: str, color: tuple, line_y: int):
    h, w = frame.shape[:2]
    
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(frame, "Detection Line", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 100), (15, 12, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"CCTV: {label}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, f"Vehicles Crossed: {state['vehicle_count']}", (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Humans Crossed  : {state['person_count']}", (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return frame


def generate_road_frames(stream_id: str, video_path: str):
    info = ROADS.get(stream_id, {"label": stream_id, "color": (255, 255, 0)})
    label = info["label"]
    color = info["color"]
    state = get_or_init_state(stream_id)

    while not os.path.exists(video_path):
        time.sleep(1.0)

    cap = cv2.VideoCapture(video_path)

    while True:
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            h, w = frame.shape[:2]
            COUNTING_LINE_Y = h // 2  

            if model is not None:
                try:
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
                    
                    if results.boxes is not None and results.boxes.id is not None:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        track_ids = results.boxes.id.int().cpu().tolist()
                        class_ids = results.boxes.cls.int().cpu().tolist()

                        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                            x1, y1, x2, y2 = box
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)

                            state["track_history"][track_id].append((cx, cy))
                            
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                            # 2. FIX: Robust Trajectory Calculation
                            if len(state["track_history"][track_id]) > 1:
                                # Compare the oldest known position to the current position
                                start_cy = state["track_history"][track_id][0][1]
                                curr_cy = state["track_history"][track_id][-1][1]

                                crossed_down = start_cy < COUNTING_LINE_Y and curr_cy >= COUNTING_LINE_Y
                                crossed_up = start_cy > COUNTING_LINE_Y and curr_cy <= COUNTING_LINE_Y

                                if (crossed_down or crossed_up) and track_id not in state["counted_ids"]:
                                    state["counted_ids"].add(track_id)
                                    
                                    # 3. FIX: Cast class_id to standard int() to prevent matching failures
                                    class_id = int(class_id)
                                    if class_id == PERSON_CLASS_ID:
                                        state["person_count"] += 1
                                    elif class_id in VEHICLE_CLASS_IDS:
                                        state["vehicle_count"] += 1
                                        
                            if len(state["track_history"][track_id]) > 20:
                                state["track_history"][track_id].pop(0)
                except Exception as e:
                    print(f"Tracking error: {e}")

            frame = _draw_overlay(frame, state, label, color, COUNTING_LINE_Y)
            
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            
            time.sleep(0.033)
        else:
            time.sleep(0.1)


def generate_yolo_frames(lot_name: str):
    """Processes the assigned video frame-by-frame with YOLO safely using defensive dictionary lookups."""
    info = PARKING.get(lot_name, {})
    TOTAL_CAPACITY = info.get("total", 500)
    
    current_occupancy = info.get("occupancy", 0)
    total_inflow = 0
    total_outflow = 0
    inflow_history = []
    outflow_history = []
    track_history = {}
    TARGET_CLASSES = [2, 3] # Car and Motorcycle

    while True:
        try:
            video_path = PARKING.get(lot_name, {}).get("video_path")
            
            # 1. Fallback Placeholder Frame if no video has been uploaded yet
            if not video_path or not os.path.exists(video_path):
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                frame[:] = (25, 20, 20) # Deep slate dark background
                
                cv2.putText(frame, "Stream Offline", (220, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (67, 168, 212), 2)
                cv2.putText(frame, "Please upload a video below to initialize YOLO tracking", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
                
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(0.1)
                continue

            # 2. Process the Active Uploaded Video Stream
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
                if PARKING.get(lot_name, {}).get("video_path") != video_path:
                    break

                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Seamless video looping
                    success, frame = cap.read()
                    if not success: 
                        time.sleep(0.5)
                        break

                h, w = frame.shape[:2]
                COUNTING_LINE_Y = int(h * 0.6) 
                current_time = time.time()

                # FIX: Access the thread lock explicitly via the model object attribute
                if model and hasattr(model, 'lock'):
                    with model.lock:
                        results = model.track(frame, persist=True, classes=TARGET_CLASSES, verbose=False)
                    
                    inflow_history = [t for t in inflow_history if current_time - t <= 60]
                    outflow_history = [t for t in outflow_history if current_time - t <= 60]

                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.int().cpu().numpy()
                        class_ids = results[0].boxes.cls.int().cpu().numpy()

                        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                            x1, y1, x2, y2 = box
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            label = "Car" if class_id == 2 else "Bike"

                            if track_id in track_history:
                                prev_cy = track_history[track_id]

                                if prev_cy < COUNTING_LINE_Y <= cy:
                                    total_inflow += 1
                                    current_occupancy = min(TOTAL_CAPACITY, current_occupancy + 1)
                                    inflow_history.append(current_time)
                                elif prev_cy > COUNTING_LINE_Y >= cy:
                                    total_outflow += 1
                                    current_occupancy = max(0, current_occupancy - 1)
                                    outflow_history.append(current_time)

                            track_history[track_id] = cy
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {track_id} {label}", (int(x1), int(y1) - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                lambda_inflow_rate = len(inflow_history)
                mu_outflow_rate = len(outflow_history)
                net_flow_rate_per_min = lambda_inflow_rate - mu_outflow_rate
                predicted_occupancy_10m = max(0, min(TOTAL_CAPACITY, current_occupancy + (10 * net_flow_rate_per_min)))

                if lot_name not in PARKING:
                    PARKING[lot_name] = {}
                
                PARKING[lot_name]["occupancy"] = current_occupancy
                PARKING[lot_name]["inflow"] = lambda_inflow_rate
                PARKING[lot_name]["outflow"] = mu_outflow_rate
                PARKING[lot_name]["predicted_occupancy"] = predicted_occupancy_10m

                predicted_pct = (predicted_occupancy_10m / TOTAL_CAPACITY) * 100 if TOTAL_CAPACITY > 0 else 0
                cv2.line(frame, (0, COUNTING_LINE_Y), (w, COUNTING_LINE_Y), (0, 0, 255), 3)
                cv2.rectangle(frame, (15, 15), (420, 140), (0, 0, 0), -1)
                cv2.putText(frame, f"Occupancy: {current_occupancy}/{TOTAL_CAPACITY}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, f"Inflow: {lambda_inflow_rate}v/m | Outflow: {mu_outflow_rate}v/m", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(frame, f"10m Forecast: {int(predicted_occupancy_10m)} ({predicted_pct:.1f}%)", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(0.03)
                
            cap.release()
            
        except Exception as stream_err:
            print(f" [CRITICAL ERROR] Core tracking engine loop broken on zone context '{lot_name}': {stream_err}")
            time.sleep(1)



# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        hashed = generate_password_hash(password)
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, email, password) VALUES (?,?,?)",
                    (username, email, hashed))
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username=?", (username,)).fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("landing"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session["username"])

@app.route("/roads")
def roads():
    return render_template("roads.html")


@app.route("/booking", methods=["GET", "POST"])
# @login_required
def booking():
    if request.method == "POST":
        pickup      = request.form.get("pickup", "")
        # Cast seats to an integer for mathematical comparison
        seats       = int(request.form.get("seats", 1)) 
        travel_date = request.form.get("travel_date", "")
        travel_time = request.form.get("travel_time", "")

        # Safeguard: Ensure no one bypasses frontend limits
        if seats > 4:
            return jsonify({"status": "error", "message": "Maximum 4 seats allowed per booking."}), 400

        with get_db() as conn:
            # POOLING ALGORITHM:
            taxi_query = """
                SELECT taxi_number, SUM(seats) as total_seats
                FROM bookings
                WHERE pickup = ? AND travel_date = ? AND travel_time = ?
                GROUP BY taxi_number
                HAVING total_seats + ? <= 4
                ORDER BY total_seats DESC
                LIMIT 1
            """
            
            # Execute search
            existing_taxi = conn.execute(
                taxi_query, 
                (pickup, travel_date, travel_time, seats)
            ).fetchone()

            if existing_taxi:
                # Pool matched! Assign the user to this existing taxi
                assigned_taxi = existing_taxi["taxi_number"]
            else:
                # No match found or all taxis are full. Allocate a new unique taxi.
                assigned_taxi = f"TAXI-{str(uuid.uuid4())[:6].upper()}"

            # Save the booking with the assigned taxi number
            conn.execute(
                "INSERT INTO bookings (user_id, pickup, seats, travel_date, travel_time, taxi_number) "
                "VALUES (?,?,?,?,?,?)",
                (session["user_id"], pickup, seats, travel_date, travel_time, assigned_taxi)
            )

        return jsonify({
            "status": "success",
            "message": f"Success! {seats} seat(s) booked from {pickup}. You are assigned to Vehicle: {assigned_taxi}."
        })

    # For GET requests: Fetch user's booking history
    history = []
    user_id = session.get("user_id")
    if user_id:
        with get_db() as conn:
            history = conn.execute("""
                SELECT u.username, b.taxi_number, b.seats, b.travel_date, b.travel_time, b.pickup 
                FROM bookings b
                JOIN users u ON b.user_id = u.id
                WHERE b.user_id = ?
                ORDER BY b.created_at DESC
            """, (user_id,)).fetchall()

    return render_template("booking.html", history=history)

@app.route("/route")
@login_required
def route_page():
    return render_template("route.html")

# ── Video streaming endpoints ──────────────────────────────────────────────

@app.route("/parking")
def parking():
    return render_template("parking.html", parking_info=PARKING)

@app.route("/api/parking_stats")
def parking_stats():
    return jsonify(PARKING)

@app.route("/upload_video/<lot_name>", methods=["POST"])
def upload_video(lot_name):
    if lot_name not in PARKING:
        return "Zone identity validation error", 404
        
    file = request.files.get("video_file")
    if file and file.filename != '':
        filename = f"{lot_name}_{int(time.time())}.mp4"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        PARKING[lot_name]["video_path"] = file_path
        print(f" [SUCCESS] Assigned video stream target for {lot_name} pointing at: {file_path}")
        
    return redirect(url_for('parking'))

@app.route("/parking_feed/<lot_name>")
def parking_feed(lot_name):
    if lot_name not in PARKING:
        return "Target endpoint sequence invalid", 404
    return Response(generate_yolo_frames(lot_name), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/<road_name>")
def video_feed(road_name):
    if road_name not in ROADS:
        return "Route invalid", 404
    vid_path = os.path.join(VIDEO_DIR, f"{road_name}.mp4")
    return Response(
        generate_road_frames(road_name, vid_path),
        mimetype="multipart/x-mixed-replace; boundary=frame")
# ── API helpers ────────────────────────────────────────────────────────────


                
    return redirect(url_for('roads'))

# FIX: Added API metrics state output route for individual asynchronous JavaScript calls
@app.route("/api/counts/<road_name>")
def api_counts(road_name):
    state = get_or_init_state(road_name)
    return jsonify({
        "vehicles": state["vehicle_count"],
        "persons": state["person_count"]
    })


# FIX: Added handling processing controller for custom multipart upload form submissions
@app.route("/upload_road_video/<road_name>", methods=["POST"])
def upload_road_video(road_name):
    if road_name not in ROADS:
        return "Invalid road target identity", 404
    
    if "video_file" not in request.files:
        return redirect(url_for('roads'))
        
    file = request.files["video_file"]
    if file.filename == "":
        return redirect(url_for('roads'))

    if file:
        dest_path = os.path.join(VIDEO_DIR, f"{road_name}.mp4")
        file.save(dest_path)
        
        with stream_lock:
            stream_state[road_name] = {
                "vehicle_count": 0,
                "person_count": 0,
                "counted_ids": set(),
                "track_history": defaultdict(list)
            }
                
    return redirect(url_for('roads'))

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(debug=True, threaded=True, port=5000, use_reloader=False)
