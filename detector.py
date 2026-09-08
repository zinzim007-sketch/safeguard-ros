"""
SGT Patrol — YOLO26 (VisDrone) Detection Server
=================================================
Runs YOLO26 on a video file (or webcam/drone feed) and broadcasts
detection results over WebSocket to the Flutter operator app.

IMPORTANT — CPU vs GPU:
On CPU-only hardware, one inference pass can genuinely take 0.5-5+
seconds. Video streaming is therefore decoupled from detection so
the camera feed stays smooth while detection runs in the background.

BEHAVIORAL LAYER:
Detections use YOLO tracking so each object can receive a persistent
track ID across frames. Track history is maintained for behavioural
analysis such as loitering.

Current behavioural capability:
- Persistent track IDs
- Position history
- Loitering detection

Future behavioural capability:
- Movement analysis
- Following
- Grouping
- Fleeing
- Suspicious interactions
- Incident scoring
- Robbery suspicion

Usage:
    python detector.py --video assets/demo/patrol_demo.mp4
    python detector.py --video 0
    python detector.py --video rtsp://your-drone-stream

The Flutter app connects to ws://localhost:8765.
"""

import asyncio
import websockets
import json
import cv2
import argparse
import base64
import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import uuid
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ultralytics import YOLO
import torch


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

PORT = 8765

MIN_CONFIDENCE = 0.4

# How often a NEW detection/inference pass is started.
DETECT_EVERY_N_FRAMES = 5

# Lower than default 640 to improve CPU performance.
INFERENCE_SIZE = 416

# Video streaming rate.
STREAM_FPS_CAP = 20

DB_PATH = 'patrol_events.db'


# -------------------------------------------------------------------------
# Behavioural layer — current loitering settings
# -------------------------------------------------------------------------

LOITER_SECONDS = 8

LOITER_RADIUS_PX = 80

LOITER_MIN_SAMPLES = 3

LOITER_ALERT_COOLDOWN = 15


print(f'[SGT] CUDA (GPU) available: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('[SGT] Running on CPU — inference will be noticeably slower.')


# -------------------------------------------------------------------------
# VisDrone classes -> SafeGuard operator labels
# -------------------------------------------------------------------------

DETECT_CLASSES = {
    'pedestrian': {
        'label': 'PERSON',
        'priority': 'high',
        'color': (0, 0, 255),
    },

    'people': {
        'label': 'PERSON',
        'priority': 'high',
        'color': (0, 0, 255),
    },

    'car': {
        'label': 'VEHICLE',
        'priority': 'medium',
        'color': (0, 165, 255),
    },

    'van': {
        'label': 'VEHICLE',
        'priority': 'medium',
        'color': (0, 165, 255),
    },

    'truck': {
        'label': 'VEHICLE',
        'priority': 'medium',
        'color': (0, 165, 255),
    },

    'bus': {
        'label': 'VEHICLE',
        'priority': 'medium',
        'color': (0, 165, 255),
    },

    'motor': {
        'label': 'VEHICLE',
        'priority': 'medium',
        'color': (0, 165, 255),
    },

    'bicycle': {
        'label': 'VEHICLE',
        'priority': 'low',
        'color': (0, 165, 255),
    },

    'tricycle': {
        'label': 'VEHICLE',
        'priority': 'low',
        'color': (0, 165, 255),
    },

    'awning-tricycle': {
        'label': 'VEHICLE',
        'priority': 'low',
        'color': (0, 165, 255),
    },
}


# Only people currently participate in loitering analysis.
PERSON_TRACK_CLASSES = {
    'pedestrian',
    'people',
}


# -------------------------------------------------------------------------
# SQLite event log
# -------------------------------------------------------------------------

_db_conn = None


def init_db():
    global _db_conn

    _db_conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

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
        '''
        INSERT INTO detection_events
        (
            id,
            timestamp,
            class_name,
            confidence,
            bbox,
            operator_action
        )
        VALUES (?, ?, ?, ?, ?, NULL)
        ''',
        (
            event_id,
            time.time(),
            class_name,
            confidence,
            json.dumps(bbox),
        )
    )

    _db_conn.commit()


def log_operator_response(event_id, response):
    _db_conn.execute(
        '''
        UPDATE detection_events
        SET
            operator_action = ?,
            operator_response_timestamp = ?
        WHERE id = ?
        ''',
        (
            response,
            time.time(),
            event_id,
        )
    )

    _db_conn.commit()

    print(
        f'[SGT] Logged operator response: '
        f'{event_id} -> {response}'
    )


class ROSDetectionPublisher(Node):

    def __init__(self):
        super().__init__('safeguard_detector')

        self.publisher = self.create_publisher(
            String,
            '/safeguard/detections',
            10
        )

        self.get_logger().info(
            'ROS publisher ready: /safeguard/detections'
        )

    def publish_detections(self, detections):
        if not detections:
            return

        msg = String()
        msg.data = json.dumps(detections)

        self.publisher.publish(msg)


# -------------------------------------------------------------------------
# Detector
# -------------------------------------------------------------------------

class SGTDetector:

    def __init__(self, video_src):

        print('[SGT] Loading YOLO26 (VisDrone) model...')

        self.model = YOLO('best.pt')

        print('[SGT] Model loaded')

        self.video_src = video_src

        self.clients = set()

        self.running = False

        self.last_detected = {}

        # Inference happens away from the asyncio event loop.
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Most recent detection boxes used for drawing.
        self._cached_boxes = []

        # Detection events waiting to be sent to Flutter.
        self._pending_new_detections = []

        # Prevent multiple inference jobs from running simultaneously.
        self._inference_in_flight = False

        # -------------------------------------------------------------
        # Behavioural state
        # -------------------------------------------------------------

        # track_id ->
        # [
        #     (timestamp, centre_x, centre_y),
        #     ...
        # ]
        self._track_history = defaultdict(list)

        # track_id -> last time loitering alert was generated
        self._loiter_last_alerted = {}

        # track_id ->
        # [
        #     {
        #         'timestamp': float,
        #         'cx': float,
        #         'cy': float
        #     },
        #     ...
        # ]
        self._movement_history = defaultdict(list)

        # Prevent movement calculations from being generated
        # for every single inference frame.
        self._last_movement_sample = {}


    # -----------------------------------------------------------------
    # Detection throttling
    # -----------------------------------------------------------------

    def _should_send(self, key):

        now = time.time()

        last = self.last_detected.get(key, 0)

        if now - last < 5:
            return False

        self.last_detected[key] = now

        return True


    # -----------------------------------------------------------------
    # Frame encoding
    # -----------------------------------------------------------------

    def _frame_to_base64(self, frame):

        _, buffer = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 60]
        )

        return base64.b64encode(buffer).decode('utf-8')


    # -----------------------------------------------------------------
    # YOLO tracking
    # -----------------------------------------------------------------

    def _run_inference(self, frame):

        """
        Run YOLO tracking.

        persist=True allows the tracker to maintain object identities
        between inference calls.

        ByteTrack assigns a track ID to detected objects.
        """

        t0 = time.time()

        result = self.model.track(
            frame,
            persist=True,
            tracker='bytetrack.yaml',
            verbose=False,
            imgsz=INFERENCE_SIZE,
        )[0]

        elapsed = time.time() - t0

        if elapsed > 0.5:
            print(
                f'[SGT] SLOW inference: '
                f'{elapsed:.2f}s for one frame — likely CPU-bound'
            )

        return result


    # -------------------------------------------------------------------------
    # Movement analysis
    # -------------------------------------------------------------------------

    MOVEMENT_HISTORY_SECONDS = 15

    MOVEMENT_MIN_DISTANCE_PX = 15

    STATIONARY_DISTANCE_PX = 25

    MOVEMENT_SAMPLE_INTERVAL = 0.5


    # -----------------------------------------------------------------
    # Behavioural layer — track history
    # -----------------------------------------------------------------

    def _update_track_history(
        self,
        track_id,
        cx,
        cy,
        now
    ):

        history = self._track_history[track_id]

        history.append(
            (
                now,
                cx,
                cy
            )
        )

        # Keep only recent history.
        cutoff = now - (LOITER_SECONDS + 5)

        while history and history[0][0] < cutoff:
            history.pop(0)

    def _update_movement_history(
        self,
        track_id,
        cx,
        cy,
        now
    ):
        """
        Store the recent movement history of a tracked object.

        Each sample records:
            timestamp
            centre X
            centre Y

        This gives the behavioural layer a memory of where
        the person/object has been moving.
        """

        last_sample = self._last_movement_sample.get(
            track_id,
            0
        )

        # Don't record essentially identical samples too frequently.
        if now - last_sample < self.MOVEMENT_SAMPLE_INTERVAL:
            return

        self._last_movement_sample[track_id] = now

        history = self._movement_history[track_id]

        history.append(
            {
                'timestamp': now,
                'cx': cx,
                'cy': cy,
            }
        )

        # Keep only the recent history window.
        cutoff = (
            now
            - self.MOVEMENT_HISTORY_SECONDS
        )

        self._movement_history[track_id] = [
            sample
            for sample in history
            if sample['timestamp'] >= cutoff
        ]


    def _calculate_movement(
        self,
        track_id
    ):
        """
        Calculate basic movement characteristics for a track.

        Returns:
            distance_px
            displacement_px
            direction
            movement_state
        """

        history = self._movement_history.get(
            track_id,
            []
        )

        if len(history) < 2:
            return {
                'distance_px': 0.0,
                'displacement_px': 0.0,
                'direction': 'unknown',
                'movement_state': 'unknown',
            }

        first = history[0]
        last = history[-1]

        # -------------------------------------------------------------
        # Total distance travelled
        # -------------------------------------------------------------

        total_distance = 0.0

        for previous, current in zip(
            history[:-1],
            history[1:]
        ):

            dx = (
                current['cx']
                - previous['cx']
            )

            dy = (
                current['cy']
                - previous['cy']
            )

            total_distance += (
                (dx ** 2 + dy ** 2)
                ** 0.5
            )

        # -------------------------------------------------------------
        # Net displacement
        # -------------------------------------------------------------

        dx = (
            last['cx']
            - first['cx']
        )

        dy = (
            last['cy']
            - first['cy']
        )

        displacement = (
            (dx ** 2 + dy ** 2)
            ** 0.5
        )

        # -------------------------------------------------------------
        # Direction
        # -------------------------------------------------------------

        if displacement < self.MOVEMENT_MIN_DISTANCE_PX:

            direction = 'stationary'

        elif abs(dx) > abs(dy):

            if dx > 0:
                direction = 'east'

            else:
                direction = 'west'

        else:

            if dy > 0:
                direction = 'south'

            else:
                direction = 'north'

        # -------------------------------------------------------------
        # Movement state
        # -------------------------------------------------------------

        if displacement <= self.STATIONARY_DISTANCE_PX:

            movement_state = 'stationary'

        else:

            movement_state = 'moving'

        return {
            'distance_px': round(
                total_distance,
                2
            ),

            'displacement_px': round(
                displacement,
                2
            ),

            'direction': direction,

            'movement_state': movement_state,
        }


    # -----------------------------------------------------------------
    # Behavioural layer — loitering
    # -----------------------------------------------------------------

    def _assess_behaviour(self, track_id: int, now: float):
        """Assess tracked movement using multiple behavioural signals.

        YOLO confidence describes object detection confidence; risk_score here
        describes behavioural evidence. They are intentionally separate.
        """
        history = self._track_history.get(track_id)
        if not history or len(history) < 3:
            return None

        samples = list(history)
        duration = samples[-1][0] - samples[0][0]
        if duration <= 0:
            return None

        positions = [(s[1], s[2]) for s in samples]
        path_distance = sum(
            math.hypot(positions[i][0] - positions[i-1][0],
                       positions[i][1] - positions[i-1][1])
            for i in range(1, len(positions))
        )
        displacement = math.hypot(
            positions[-1][0] - positions[0][0],
            positions[-1][1] - positions[0][1],
        )
        average_speed = path_distance / max(duration, 0.001)
        path_efficiency = displacement / path_distance if path_distance > 1.0 else 1.0

        stationary_segments = 0
        total_segments = max(len(positions) - 1, 1)
        for i in range(1, len(positions)):
            if math.hypot(positions[i][0] - positions[i-1][0],
                          positions[i][1] - positions[i-1][1]) <= 12.0:
                stationary_segments += 1
        stationary_ratio = stationary_segments / total_segments

        loitering_score = 0
        if duration >= 20.0:
            loitering_score += 25
        elif duration >= 12.0:
            loitering_score += 10
        if stationary_ratio >= 0.75:
            loitering_score += 35
        elif stationary_ratio >= 0.60:
            loitering_score += 20
        elif stationary_ratio >= 0.45:
            loitering_score += 10
        if average_speed <= 4.0:
            loitering_score += 25
        elif average_speed <= 7.0:
            loitering_score += 12
        if displacement <= 45.0:
            loitering_score += 15
        elif displacement <= 80.0:
            loitering_score += 8

        circulating_score = 0
        if duration >= 25.0:
            circulating_score += 15
        if path_distance >= 150.0:
            circulating_score += 25
        elif path_distance >= 90.0:
            circulating_score += 15
        if path_efficiency <= 0.35:
            circulating_score += 35
        elif path_efficiency <= 0.55:
            circulating_score += 20
        if stationary_ratio < 0.65:
            circulating_score += 10

        if loitering_score >= 65:
            behaviour = 'possible_loitering'
            risk_score = loitering_score
        elif circulating_score >= 65:
            behaviour = 'circulating'
            risk_score = circulating_score
        else:
            behaviour = 'normal'
            risk_score = max(loitering_score, circulating_score)

        risk_score = min(int(risk_score), 100)
        if risk_score >= 80:
            priority = 'high'
        elif risk_score >= 60:
            priority = 'medium'
        elif risk_score >= 35:
            priority = 'watch'
        else:
            priority = 'low'

        return {
            'behaviour': behaviour,
            'risk_score': risk_score,
            'priority': priority,
            'duration_s': round(duration, 1),
            'path_distance_px': round(path_distance, 2),
            'displacement_px': round(displacement, 2),
            'average_speed_px_s': round(average_speed, 2),
            'stationary_ratio': round(stationary_ratio, 2),
            'path_efficiency': round(path_efficiency, 2),
        }

    def _check_loitering(self, track_id: int, now: float):
        """Compatibility wrapper used by the existing detection pipeline."""
        assessment = self._assess_behaviour(track_id, now)
        if not assessment or assessment['behaviour'] != 'possible_loitering':
            return None

        last_event = self._loiter_last_alerted.get(track_id, 0.0)
        if now - last_event < 10.0:
            return None
        self._loiter_last_alerted[track_id] = now

        print(
            '[SGT] LOITERING flagged — '
            f'track #{track_id}, '
            f"risk={assessment['risk_score']}, "
            f"priority={assessment['priority']}, "
            f"duration={assessment['duration_s']}s, "
            f"stationary={assessment['stationary_ratio']:.2f}"
        )

        return {
            'class': 'loitering',
            'label': 'POSSIBLE_LOITERING',
            'confidence': assessment['risk_score'],
            'priority': assessment['priority'],
            'track_id': track_id,
            'behaviour': assessment,
            'timestamp': now,
        }

    def _should_alert_loitering(
        self,
        track_id,
        now
    ):

        last = self._loiter_last_alerted.get(
            track_id,
            0
        )

        if now - last < LOITER_ALERT_COOLDOWN:
            return False

        self._loiter_last_alerted[track_id] = now

        return True


    # -----------------------------------------------------------------
    # Background detection
    # -----------------------------------------------------------------

    async def _detect_async(self, frame):

        """
        Perform detection in a background thread.

        The video/WebSocket loop is therefore not blocked by inference.
        """

        self._inference_in_flight = True

        try:

            loop = asyncio.get_event_loop()

            results = await loop.run_in_executor(
                self._executor,
                self._run_inference,
                frame
            )

            boxes = []

            new_detections = []

            now = time.time()


            # ---------------------------------------------------------
            # Process every detected object
            # ---------------------------------------------------------

            for box in results.boxes:

                class_id = int(
                    box.cls[0]
                )

                class_name = self.model.names[
                    class_id
                ]

                confidence = float(
                    box.conf[0]
                )


                # Ignore classes that SafeGuard doesn't use.
                if class_name not in DETECT_CLASSES:
                    continue


                # Ignore low-confidence detections.
                if confidence < MIN_CONFIDENCE:
                    continue


                info = DETECT_CLASSES[
                    class_name
                ]


                # Bounding box.
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                # -----------------------------------------------------
                # Persistent tracking ID
                # -----------------------------------------------------

                track_id = (
                    int(box.id[0])
                    if box.id is not None
                    else None
                )


                # -----------------------------------------------------
                # Behaviour analysis
                # -----------------------------------------------------

                is_loitering = False
                movement = None


                if (
                    track_id is not None
                    and
                    class_name in PERSON_TRACK_CLASSES
                ):

                    cx = (
                        x1 + x2
                    ) / 2

                    cy = (
                        y1 + y2
                    ) / 2

                    # Existing loitering history.
                    self._update_track_history(
                        track_id,
                        cx,
                        cy,
                        now
                    )

                    # New movement history.
                    self._update_movement_history(
                        track_id,
                        cx,
                        cy,
                        now
                    )

                    # Calculate current movement.
                    movement = self._calculate_movement(
                        track_id
                    )

                    is_loitering = (
                        self._check_loitering(
                            track_id,
                            now
                        )
                    )





                # -----------------------------------------------------
                # Draw detection
                # -----------------------------------------------------

                boxes.append(
                    {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,

                        'color': (
                            (0, 255, 255)
                            if is_loitering
                            else info['color']
                        ),

                        'label_text': (
                            f'LOITERING #{track_id}'
                            if is_loitering
                            else (
                                f"{info['label']} "
                                f"{int(confidence * 100)}%"
                            )
                        ),
                    }
                )


                # -----------------------------------------------------
                # Normal detection event
                # -----------------------------------------------------

                throttle_key = info['label']


                if self._should_send(
                    throttle_key
                ):

                    event_id = str(
                        uuid.uuid4()
                    )

                    bbox = [
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1
                    ]


                    log_detection(
                        event_id,
                        class_name,
                        int(confidence * 100),
                        bbox
                    )


                    # IMPORTANT:
                    # Normal detections now expose track_id.
                    #
                    # This means downstream systems can tell that
                    # multiple observations belong to the same
                    # tracked person/vehicle.

                    new_detections.append(
                        {
                            'id': event_id,

                            'track_id': track_id,

                            'class': class_name,
                            'movement': movement if track_id is not None else None,

                            'label': info['label'],

                            'confidence': int(
                                confidence * 100
                            ),

                            'priority': info['priority'],

                            'bbox': bbox,

                            'timestamp': int(
                                time.time() * 1000
                            ),
                        }
                    )


                # -----------------------------------------------------
                # Loitering event
                # -----------------------------------------------------

                if is_loitering:

                    event_id = str(
                        uuid.uuid4()
                    )

                    bbox = [
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1
                    ]


                    log_detection(
                        event_id,
                        'loitering',
                        int(confidence * 100),
                        bbox
                    )


                    new_detections.append(
                        {
                            'id': event_id,

                            'class': 'loitering',

                            'label': 'LOITERING',

                            'confidence': int(
                                confidence * 100
                            ),

                            'priority': 'critical',

                            'bbox': bbox,

                            'timestamp': int(
                                time.time() * 1000
                            ),

                            'track_id': track_id,
                        }
                    )


                    print(
                        '[SGT] LOITERING flagged — '
                        f'track #{track_id}, '
                        f'{span_desc(self._track_history[track_id])}'
                    )


            # ---------------------------------------------------------
            # Update shared detector state
            # ---------------------------------------------------------

            self._cached_boxes = boxes

            self._pending_new_detections.extend(
                new_detections
            )

            if new_detections:
                self.ros_node.publish_detections(
                    new_detections
                )

                rclpy.spin_once(
                    self.ros_node,
                    timeout_sec=0
                )


        finally:

            self._inference_in_flight = False


    # -----------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------

    def _draw_cached_boxes(self, frame):

        """
        Cheap drawing operation.

        Runs on every streamed frame.
        Does NOT perform inference.
        """

        for b in self._cached_boxes:

            x1 = b['x1']
            y1 = b['y1']
            x2 = b['x2']
            y2 = b['y2']

            color = b['color']


            # Main rectangle.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            # Corner brackets.
            br = 12


            cv2.line(
                frame,
                (x1, y1),
                (x1 + br, y1),
                color,
                2
            )

            cv2.line(
                frame,
                (x1, y1),
                (x1, y1 + br),
                color,
                2
            )


            cv2.line(
                frame,
                (x2, y1),
                (x2 - br, y1),
                color,
                2
            )

            cv2.line(
                frame,
                (x2, y1),
                (x2, y1 + br),
                color,
                2
            )


            cv2.line(
                frame,
                (x1, y2),
                (x1 + br, y2),
                color,
                2
            )

            cv2.line(
                frame,
                (x1, y2),
                (x1, y2 - br),
                color,
                2
            )


            cv2.line(
                frame,
                (x2, y2),
                (x2 - br, y2),
                color,
                2
            )

            cv2.line(
                frame,
                (x2, y2),
                (x2, y2 - br),
                color,
                2
            )


            # Label.
            label = b['label_text']


            cv2.rectangle(
                frame,
                (
                    x1,
                    y1 - 20
                ),
                (
                    x1 + len(label) * 8,
                    y1
                ),
                color,
                -1
            )


            cv2.putText(
                frame,
                label,
                (
                    x1 + 2,
                    y1 - 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


        return frame


    # -----------------------------------------------------------------
    # Main video loop
    # -----------------------------------------------------------------

    async def run(self):

        self.running = True


        # Numeric camera IDs become integers.
        src = (
            int(self.video_src)
            if self.video_src.isdigit()
            else self.video_src
        )


        cap = cv2.VideoCapture(src)


        if not cap.isOpened():

            print(
                '[SGT] ERROR: '
                f'Could not open video: {self.video_src}'
            )

            return


        print(
            f'[SGT] Detection running on: '
            f'{self.video_src}'
        )


        frame_count = 0

        min_frame_interval = (
            1.0 / STREAM_FPS_CAP
        )


        while self.running:

            loop_start = time.time()


            ret, frame = cap.read()


            if not ret:

                # Loop recorded videos.
                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                continue


            frame_count += 1


            # ---------------------------------------------------------
            # Start background inference periodically.
            # ---------------------------------------------------------

            if (
                frame_count % DETECT_EVERY_N_FRAMES == 0
                and
                not self._inference_in_flight
            ):

                asyncio.create_task(
                    self._detect_async(
                        frame.copy()
                    )
                )


            # ---------------------------------------------------------
            # Draw most recent detection results.
            # ---------------------------------------------------------

            annotated_frame = (
                self._draw_cached_boxes(
                    frame.copy()
                )
            )


            # ---------------------------------------------------------
            # Send to Flutter.
            # ---------------------------------------------------------

            if self.clients:

                frame_b64 = (
                    self._frame_to_base64(
                        annotated_frame
                    )
                )


                detections = (
                    self._pending_new_detections
                )


                # Only send each new detection once.
                self._pending_new_detections = []


                message = json.dumps(
                    {
                        'type': 'frame',

                        'frame': frame_b64,

                        'detections': detections,
                    }
                )


                disconnected = set()


                for client in list(self.clients):

                    try:

                        await client.send(
                            message
                        )

                    except websockets.exceptions.ConnectionClosed:

                        disconnected.add(
                            client
                        )


                self.clients -= disconnected


            # ---------------------------------------------------------
            # Keep streaming at target FPS.
            # ---------------------------------------------------------

            elapsed = (
                time.time()
                - loop_start
            )


            await asyncio.sleep(
                max(
                    0.0,
                    min_frame_interval - elapsed
                )
            )


        cap.release()

        print(
            '[SGT] Detection stopped'
        )


    def stop(self):

        self.running = False


# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------

def span_desc(history):

    if not history:
        return 'no history'


    return (
        f'{len(history)} samples over '
        f'{history[-1][0] - history[0][0]:.1f}s'
    )


# -------------------------------------------------------------------------
# WebSocket server
# -------------------------------------------------------------------------

detector = None


async def handle_client(websocket):

    print(
        f'[SGT] Client connected: '
        f'{websocket.remote_address}'
    )


    detector.clients.add(
        websocket
    )


    try:

        async for message in websocket:

            try:

                cmd = json.loads(
                    message
                )

                action = cmd.get(
                    'action'
                )


                # -----------------------------------------------------
                # Stop detector
                # -----------------------------------------------------

                if action == 'stop':

                    detector.stop()


                # -----------------------------------------------------
                # Operator response
                # -----------------------------------------------------

                elif action == 'operator_response':

                    event_id = cmd.get(
                        'id'
                    )

                    response = cmd.get(
                        'response'
                    )


                    if (
                        event_id
                        and
                        response in (
                            'confirmed',
                            'dismissed'
                        )
                    ):

                        log_operator_response(
                            event_id,
                            response
                        )


            except json.JSONDecodeError:

                pass


    except websockets.exceptions.ConnectionClosed:

        pass


    finally:

        detector.clients.discard(
            websocket
        )


        print(
            f'[SGT] Client disconnected: '
            f'{websocket.remote_address}'
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

async def main(video_src):

    global detector

    init_db()

    # Start ROS 2
    rclpy.init()

    ros_node = ROSDetectionPublisher()

    detector = SGTDetector(
        video_src
    )

    detector.ros_node = ros_node



    print(
        f'[SGT] Starting WebSocket server '
        f'on ws://localhost:{PORT}'
    )


    async with websockets.serve(
        handle_client,
        'localhost',
        PORT
    ):

        await detector.run()
        ros_node.destroy_node()
        rclpy.shutdown()


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='SGT Patrol Detection Server'
    )


    parser.add_argument(
        '--video',
        default='0',
        help=(
            'Video source: path to mp4, '
            '0 for webcam, or RTSP URL'
        )
    )


    args = parser.parse_args()


    try:

        asyncio.run(
            main(args.video)
        )

    except KeyboardInterrupt:

        print(
            '\n[SGT] Server stopped'
        )