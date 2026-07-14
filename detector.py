"""
SGT Patrol — YOLO26 (VisDrone) Detection Server
=================================================
Runs YOLO26 on a video file (or webcam/drone feed) and broadcasts
detection results over WebSocket to the Flutter operator app.

Every detection is logged to a local SQLite event log. When the
operator confirms or dismisses an alert in the app, that decision is
sent back over the same WebSocket and written to the matching row.

Performance notes (read this if the feed looks slow/paused):
- Model inference runs in a background thread via run_in_executor,
  so it no longer blocks the WebSocket/frame loop while it computes.
- The SQLite connection is opened once at startup and reused, instead
  of opening/closing a new connection on every detection.
- MIN_CONFIDENCE and DETECT_EVERY_N_FRAMES below are your two main
  speed/accuracy knobs if it's still too slow on your machine.

Usage:
    python detector.py --video assets/demo/patrol_demo.mp4
    python detector.py --video 0                              (webcam)
    python detector.py --video rtsp://your-drone-stream       (live feed)

The Flutter app connects to ws://localhost:8765 and receives JSON
detection events in real time.
"""

import asyncio
import websockets
import json
import cv2
import argparse
import base64
import time
import uuid
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import String

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

PORT = 8765
MIN_CONFIDENCE = 0.4          # raise this (e.g. 0.5) if you're getting noisy/low-quality detections
DETECT_EVERY_N_FRAMES = 5     # raise this (e.g. 8-10) if the feed is still slow — trades detection latency for speed
INFERENCE_SIZE = 416          # lower than default 640 — smaller = faster on CPU, some accuracy tradeoff
DB_PATH = 'patrol_events.db'

print(f'[SGT] CUDA (GPU) available: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('[SGT] Running on CPU — inference will be noticeably slower than on Kaggle\'s GPU.')

# VisDrone class names -> operator-facing labels
DETECT_CLASSES = {
    'pedestrian':      {'label': 'PERSON',  'priority': 'high',   'color': (0, 0, 255)},
    'people':          {'label': 'PERSON',  'priority': 'high',   'color': (0, 0, 255)},
    'car':             {'label': 'VEHICLE', 'priority': 'medium', 'color': (0, 165, 255)},
    'van':             {'label': 'VEHICLE', 'priority': 'medium', 'color': (0, 165, 255)},
    'truck':           {'label': 'VEHICLE', 'priority': 'medium', 'color': (0, 165, 255)},
    'bus':             {'label': 'VEHICLE', 'priority': 'medium', 'color': (0, 165, 255)},
    'motor':           {'label': 'VEHICLE', 'priority': 'medium', 'color': (0, 165, 255)},
    'bicycle':         {'label': 'VEHICLE', 'priority': 'low',    'color': (0, 165, 255)},
    'tricycle':        {'label': 'VEHICLE', 'priority': 'low',    'color': (0, 165, 255)},
    'awning-tricycle': {'label': 'VEHICLE', 'priority': 'low',    'color': (0, 165, 255)},
}

# -------------------------------------------------------------------------
# Event log (SQLite) — one persistent connection, opened once.
# -------------------------------------------------------------------------

_db_conn = None

def init_db():
    global _db_conn
    _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _db_conn.execute('''
        CREATE TABLE IF NOT EXISTS detection_events (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            class_name TEXT,
            confidence INTEGER,
            bbox TEXT,
            operator_action TEXT,
            operator_response_timestamp REAL
        )
    ''')
    _db_conn.commit()
    print(f'[SGT] Event log ready at {DB_PATH}')


def log_detection(event_id, class_name, confidence, bbox):
    _db_conn.execute(
        '''INSERT INTO detection_events
           (id, timestamp, class_name, confidence, bbox, operator_action)
           VALUES (?, ?, ?, ?, ?, NULL)''',
        (event_id, time.time(), class_name, confidence, json.dumps(bbox))
    )
    _db_conn.commit()


def log_operator_response(event_id, response):
    _db_conn.execute(
        '''UPDATE detection_events
           SET operator_action = ?, operator_response_timestamp = ?
           WHERE id = ?''',
        (response, time.time(), event_id)
    )
    _db_conn.commit()
    print(f'[SGT] Logged operator response: {event_id} -> {response}')


# -------------------------------------------------------------------------
# Detector
# -------------------------------------------------------------------------

class SGTDetector:
    def __init__(self, video_src):
        print('[SGT] Loading YOLO26 (VisDrone) model...')
        self.model = YOLO('best.pt')  # rename to match your actual downloaded weights file
        print('[SGT] Model loaded')

        self.video_src = video_src
        self.clients = set()
        self.running = False
        self.last_detected = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        rclpy.init()
        self._ros_node = Node('sgt_detector')
        self._ros_pub = self._ros_node.create_publisher(String, '/safeguard/detections', 10)

    def _should_send(self, key):
        now = time.time()
        last = self.last_detected.get(key, 0)
        if now - last < 5:
            return False
        self.last_detected[key] = now
        return True

    def _frame_to_base64(self, frame):
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return base64.b64encode(buffer).decode('utf-8')

    def _run_inference(self, frame):
        """Runs in a background thread via run_in_executor — this is the
        blocking call, kept OFF the asyncio event loop so frame sending
        and WebSocket messages keep flowing while it computes."""
        t0 = time.time()
        result = self.model(frame, verbose=False, imgsz=INFERENCE_SIZE)[0]
        elapsed = time.time() - t0
        if elapsed > 0.5:
            print(f'[SGT] SLOW inference: {elapsed:.2f}s for one frame — likely CPU-bound')
        return result

    async def run(self):
        self.running = True
        src = int(self.video_src) if self.video_src == '0' else self.video_src
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f'[SGT] ERROR: Could not open video: {self.video_src}')
            return

        print(f'[SGT] Detection running on: {self.video_src}')
        frame_count = 0
        loop = asyncio.get_event_loop()

        while self.running:
            ret, frame = cap.read()

            if not ret:
                # Loop video for demo
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1

            if frame_count % DETECT_EVERY_N_FRAMES != 0:
                await asyncio.sleep(0.01)
                continue

            # Non-blocking — inference happens in a background thread,
            # the event loop stays free to send frames/handle messages
            # while this runs.
            results = await loop.run_in_executor(self._executor, self._run_inference, frame)

            detections = []
            annotated_frame = frame.copy()

            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                if class_name not in DETECT_CLASSES:
                    continue
                if confidence < MIN_CONFIDENCE:
                    continue

                info = DETECT_CLASSES[class_name]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = info['color']

                # Bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Corner brackets
                b = 12
                cv2.line(annotated_frame, (x1, y1), (x1 + b, y1), color, 2)
                cv2.line(annotated_frame, (x1, y1), (x1, y1 + b), color, 2)
                cv2.line(annotated_frame, (x2, y1), (x2 - b, y1), color, 2)
                cv2.line(annotated_frame, (x2, y1), (x2, y1 + b), color, 2)
                cv2.line(annotated_frame, (x1, y2), (x1 + b, y2), color, 2)
                cv2.line(annotated_frame, (x1, y2), (x1, y2 - b), color, 2)
                cv2.line(annotated_frame, (x2, y2), (x2 - b, y2), color, 2)
                cv2.line(annotated_frame, (x2, y2), (x2, y2 - b), color, 2)

                # Label
                label = f"{info['label']} {int(confidence * 100)}%"
                cv2.rectangle(annotated_frame, (x1, y1 - 20),
                    (x1 + len(label) * 8, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Throttle by LABEL (PERSON/VEHICLE), not raw class name —
                # this is the fix for multiple vehicle subtypes (car, van,
                # truck...) each firing their own separate alert in the
                # same busy frame. All vehicle subtypes now share one
                # 5-second window.
                throttle_key = info['label']
                if self._should_send(throttle_key):
                    event_id = str(uuid.uuid4())
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    log_detection(event_id, class_name, int(confidence * 100), bbox)

                    detections.append({
                        'id': event_id,
                        'class': class_name,
                        'label': info['label'],
                        'confidence': int(confidence * 100),
                        'priority': info['priority'],
                        'bbox': bbox,
                        'timestamp': int(time.time() * 1000),
                    })

            if self.clients:
                frame_b64 = self._frame_to_base64(annotated_frame)
                message = json.dumps({
                    'type': 'frame',
                    'frame': frame_b64,
                    'detections': detections,
                })

                disconnected = set()
                for client in self.clients:
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.add(client)
                self.clients -= disconnected

            # Publish detections to ROS 2
            if detections:
                ros_msg = String()
                ros_msg.data = json.dumps(detections)
                self._ros_pub.publish(ros_msg)

            await asyncio.sleep(0.03)

        cap.release()
        print('[SGT] Detection stopped')

    def stop(self):
        self.running = False
        self._ros_node.destroy_node()
        rclpy.shutdown()


# -------------------------------------------------------------------------
# WebSocket server
# -------------------------------------------------------------------------

detector = None

async def handle_client(websocket):
    print(f'[SGT] Client connected: {websocket.remote_address}')
    detector.clients.add(websocket)
    try:
        async for message in websocket:
            try:
                cmd = json.loads(message)
                action = cmd.get('action')

                if action == 'stop':
                    detector.stop()

                elif action == 'operator_response':
                    event_id = cmd.get('id')
                    response = cmd.get('response')
                    if event_id and response in ('confirmed', 'dismissed'):
                        log_operator_response(event_id, response)

            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        detector.clients.discard(websocket)
        print(f'[SGT] Client disconnected: {websocket.remote_address}')


async def main(video_src):
    global detector
    init_db()
    detector = SGTDetector(video_src)

    print(f'[SGT] Starting WebSocket server on ws://localhost:{PORT}')

    async with websockets.serve(handle_client, 'localhost', PORT):
        await detector.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGT Patrol Detection Server')
    parser.add_argument('--video', default='0',
        help='Video source: path to mp4, 0 for webcam, or RTSP URL')
    args = parser.parse_args()

    try:
        asyncio.run(main(args.video))
    except KeyboardInterrupt:
        print('\n[SGT] Server stopped')