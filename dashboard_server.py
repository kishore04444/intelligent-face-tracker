import os
import sqlite3
import threading
import time
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, send_from_directory


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DASHBOARD_DIR = os.path.join(
    BASE_DIR,
    "dashboard"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "visitors.db"
)


# ============================================================
# CURRENT SESSION
# ============================================================

# Every time main.py starts, this module is imported again.
# Therefore this timestamp represents the beginning of the
# current tracker session.

SESSION_START = datetime.now().isoformat(
    timespec="microseconds"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    static_folder=DASHBOARD_DIR,
    static_url_path=""
)


# ============================================================
# LIVE FRAME
# ============================================================

latest_frame = None

frame_lock = threading.Lock()


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.route("/")
def dashboard():

    index_path = os.path.join(
        DASHBOARD_DIR,
        "index.html"
    )

    if not os.path.exists(index_path):

        return (
            "Dashboard index.html not found.<br><br>"
            f"Expected location:<br>{index_path}",
            404
        )

    return send_from_directory(
        DASHBOARD_DIR,
        "index.html"
    )


# ============================================================
# STATIC FILES
# ============================================================

@app.route("/<path:filename>")
def static_files(filename):

    file_path = os.path.join(
        DASHBOARD_DIR,
        filename
    )

    if os.path.isfile(file_path):

        return send_from_directory(
            DASHBOARD_DIR,
            filename
        )

    return "Not Found", 404


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=5
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# GET CURRENT SESSION EVENTS
# ============================================================

def get_session_events():

    if not os.path.exists(DATABASE_PATH):

        return []

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                visitor_id,
                event_type,
                timestamp,
                image_path
            FROM events
            WHERE timestamp >= ?
            ORDER BY id DESC
            """,
            (SESSION_START,)
        )

        rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "visitor_id": row["visitor_id"],
                "event_type": row["event_type"],
                "timestamp": row["timestamp"],
                "image_path": row["image_path"]
            }
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# API - STATS
# ============================================================

@app.route("/api/stats")
def stats():

    try:

        events = get_session_events()

        # ----------------------------------------------------
        # Current session ENTRY events
        # ----------------------------------------------------

        entry_events = [
            event
            for event in events
            if event["event_type"] == "ENTRY"
        ]

        # ----------------------------------------------------
        # Current session EXIT events
        # ----------------------------------------------------

        exit_events = [
            event
            for event in events
            if event["event_type"] == "EXIT"
        ]

        # ----------------------------------------------------
        # Unique visitors during THIS session
        # ----------------------------------------------------

        unique_visitor_ids = {
            event["visitor_id"]
            for event in entry_events
            if event["visitor_id"] is not None
        }

        # ----------------------------------------------------
        # Determine currently active visitors
        #
        # Process events from oldest -> newest.
        # ENTRY = active
        # EXIT  = inactive
        # ----------------------------------------------------

        visitor_state = {}

        for event in reversed(events):

            visitor_id = event["visitor_id"]

            if visitor_id is None:
                continue

            if event["event_type"] == "ENTRY":

                visitor_state[visitor_id] = True

            elif event["event_type"] == "EXIT":

                visitor_state[visitor_id] = False

        active_visitors = sum(
            1
            for active in visitor_state.values()
            if active
        )

        return jsonify({

            # Current session only
            "unique_visitors": len(unique_visitor_ids),

            "active_visitors": active_visitors,

            "entries": len(entry_events),

            "exits": len(exit_events),

            # Compatibility aliases
            "unique": len(unique_visitor_ids),

            "active": active_visitors,

            "session_start": SESSION_START
        })

    except Exception as error:

        return jsonify({

            "unique_visitors": 0,

            "active_visitors": 0,

            "entries": 0,

            "exits": 0,

            "unique": 0,

            "active": 0,

            "error": str(error)

        }), 500


# ============================================================
# API - EVENTS
# ============================================================

@app.route("/api/events")
def events():

    try:

        return jsonify(
            get_session_events()
        )

    except Exception as error:

        return jsonify({
            "error": str(error),
            "events": []
        }), 500


# ============================================================
# API - ALL REGISTERED VISITORS
#
# This intentionally shows the complete visitor database.
# The statistics above are session-specific.
# ============================================================

@app.route("/api/visitors")
def visitors():

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                first_seen,
                last_seen
            FROM visitors
            ORDER BY id DESC
            """
        )

        rows = cursor.fetchall()

        connection.close()

        result = []

        for row in rows:

            result.append({

                "id": row["id"],

                "visitor_id": row["id"],

                "first_seen": row["first_seen"],

                "last_seen": row["last_seen"]

            })

        return jsonify(result)

    except Exception as error:

        return jsonify({
            "error": str(error),
            "visitors": []
        }), 500


# ============================================================
# LIVE CAMERA FRAME
# ============================================================

def update_frame(frame):

    global latest_frame

    if frame is None:
        return

    try:

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
        )

        if not success:
            return

        with frame_lock:

            latest_frame = encoded.tobytes()

    except Exception:

        pass


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_stream():

    while True:

        with frame_lock:

            frame = latest_frame

        if frame is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(0.03)


@app.route("/api/stream")
def stream():

    return Response(
        generate_stream(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# DASHBOARD SERVER CLASS
# ============================================================

class DashboardServer:

    def __init__(
        self,
        host="127.0.0.1",
        port=5000
    ):

        self.host = host
        self.port = port
        self.thread = None

    def start(self):

        if self.thread is not None:
            return

        def run_server():

            app.run(
                host=self.host,
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False
            )

        self.thread = threading.Thread(
            target=run_server,
            daemon=True
        )

        self.thread.start()

        print(
            f"Dashboard: http://{self.host}:{self.port}"
        )

        print(
            f"Dashboard session started: {SESSION_START}"
        )

    def update_frame(self, frame):

        update_frame(frame)

    def stop(self):

        # Flask runs in a daemon thread.
        # It will terminate automatically when main.py exits.

        self.thread = None


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print(
        "=============================================="
    )

    print(
        "Intelligent Visitor Analytics Dashboard"
    )

    print(
        "=============================================="
    )

    print(
        f"Dashboard folder: {DASHBOARD_DIR}"
    )

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        f"Session started: {SESSION_START}"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )