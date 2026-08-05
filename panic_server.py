"""
SGT Patrol — Panic Button Demo Server
======================================
Runs a simple HTTP server that:
1. Serves a panic button web page to any phone on the network
2. Receives GPS coordinates when the button is pressed
3. Publishes to ROS 2 /safeguard/panic topic
4. Mission planner picks it up and dispatches the drone

Usage:
    python3 panic_server.py

Then on any phone on the same WiFi:
    Open http://<your-laptop-ip>:5000
    Press the panic button
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# -------------------------------------------------------------------------
# ROS 2 panic publisher node
# -------------------------------------------------------------------------

class PanicPublisher(Node):
    def __init__(self):
        super().__init__('panic_button')
        self.publisher = self.create_publisher(String, '/safeguard/panic', 10)
        self.get_logger().info('Panic button server ready')

    def publish_panic(self, lat, lng, device_id='unknown'):
        msg = String()
        msg.data = json.dumps({
            'type': 'panic',
            'latitude': lat,
            'longitude': lng,
            'device_id': device_id,
            'timestamp': __import__('time').time() * 1000,
        })
        self.publisher.publish(msg)
        self.get_logger().info(f'PANIC published: {lat}, {lng} from {device_id}')

ros_node = None

# -------------------------------------------------------------------------
# HTTP server — serves the panic button web page and receives GPS
# -------------------------------------------------------------------------

PANIC_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SGT Patrol — Emergency</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #070f1a;
      color: #a8c8f0;
      font-family: -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 24px;
    }
    .logo {
      font-size: 11px;
      letter-spacing: 3px;
      color: #4a7fc1;
      margin-bottom: 48px;
    }
    .status {
      font-size: 13px;
      color: #4a6a8a;
      margin-bottom: 32px;
      text-align: center;
      min-height: 20px;
    }
    .panic-btn {
      width: 200px;
      height: 200px;
      border-radius: 50%;
      background: #2a0a0a;
      border: 3px solid #e74c3c;
      color: #e74c3c;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 2px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s;
      -webkit-tap-highlight-color: transparent;
    }
    .panic-btn:active {
      background: #e74c3c;
      color: white;
      transform: scale(0.95);
    }
    .panic-btn.sent {
      background: #0d2a1a;
      border-color: #2ecc71;
      color: #2ecc71;
    }
    .icon { font-size: 32px; }
    .note {
      margin-top: 48px;
      font-size: 11px;
      color: #2a4a6a;
      text-align: center;
      max-width: 280px;
      line-height: 1.6;
    }
  </style>
</head>
<body>
  <div class="logo">SGT PATROL</div>
  <div class="status" id="status">Getting your location...</div>
  <button class="panic-btn" id="btn" onclick="sendPanic()">
    <span class="icon">🆘</span>
    <span>EMERGENCY</span>
  </button>
  <div class="note">
    Press and hold to send your location to the nearest patrol drone.
    Help will be dispatched immediately.
  </div>

  <script>
    let userLat = null;
    let userLng = null;

    // Get GPS as soon as page loads
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        pos => {
          userLat = pos.coords.latitude;
          userLng = pos.coords.longitude;
          document.getElementById('status').textContent =
            'Location ready. Press button in emergency.';
        },
        err => {
          document.getElementById('status').textContent =
            'Location unavailable. Press button to alert without GPS.';
        },
        { enableHighAccuracy: true, timeout: 10000 }
      );
    }

    function sendPanic() {
      const btn = document.getElementById('btn');
      const status = document.getElementById('status');

      fetch('/panic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: userLat,
          longitude: userLng,
          device: navigator.userAgent.substring(0, 50),
        })
      })
      .then(r => r.json())
      .then(data => {
        btn.classList.add('sent');
        btn.innerHTML = '<span class="icon">✓</span><span>HELP COMING</span>';
        status.textContent = 'Drone dispatched to your location.';
      })
      .catch(() => {
        status.textContent = 'Error sending alert. Try again.';
      });
    }
  </script>
</body>
</html>"""


class PanicHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default HTTP logs

    def do_GET(self):
        if self.path == '/' or self.path == '/panic':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(PANIC_PAGE.encode())

    def do_POST(self):
        if self.path == '/panic':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))

            lat = body.get('latitude') or 0.0
            lng = body.get('longitude') or 0.0
            device = body.get('device', 'unknown')

            print(f'[PANIC] Received from {device}: lat={lat}, lng={lng}')

            # Publish to ROS 2
            if ros_node:
                ros_node.publish_panic(lat, lng, device)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'dispatched'}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def run_http_server():
    server = HTTPServer(('0.0.0.0', 5000), PanicHandler)
    print('[PANIC] HTTP server on http://0.0.0.0:5000')
    print('[PANIC] Open this on your phone: http://<your-laptop-ip>:5000')
    server.serve_forever()


if __name__ == '__main__':
    # Start ROS 2
    rclpy.init()
    ros_node = PanicPublisher()

    # Run HTTP server in background thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # Spin ROS 2
    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        print('\n[PANIC] Server stopped')
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
