"""
SGT Patrol — Panic ROS 2 Bridge (WSL)
======================================
Listens for UDP panic events from Windows panic server
and publishes them to ROS 2 /safeguard/panic topic.

Usage (in Ubuntu):
    source /opt/ros/humble/setup.bash
    python3 ~/safeguard_ws/panic_ros_bridge.py
"""

import socket
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading


class PanicRosBridge(Node):
    def __init__(self):
        super().__init__('panic_ros_bridge')
        self.publisher = self.create_publisher(String, '/safeguard/panic', 10)
        self.get_logger().info('Panic ROS 2 bridge ready')

    def publish_panic(self, data):
        msg = String()
        msg.data = json.dumps(data)
        self.publisher.publish(msg)
        self.get_logger().info(
            f'PANIC published to ROS 2: lat={data.get("latitude")}, lng={data.get("longitude")}')


def udp_listener(ros_node):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 5001))
    print('[PANIC BRIDGE] Listening for panic events on UDP port 5001')

    while True:
        data, addr = sock.recvfrom(1024)
        try:
            payload = json.loads(data.decode())
            print(f'[PANIC BRIDGE] Received from {addr}: {payload}')
            ros_node.publish_panic(payload)
        except Exception as e:
            print(f'[PANIC BRIDGE] Error: {e}')


if __name__ == '__main__':
    rclpy.init()
    node = PanicRosBridge()

    udp_thread = threading.Thread(target=udp_listener, args=(node,), daemon=True)
    udp_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n[PANIC BRIDGE] Stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()
