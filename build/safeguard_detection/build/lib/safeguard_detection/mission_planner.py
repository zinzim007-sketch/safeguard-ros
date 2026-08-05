import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class MissionPlanner(Node):

    def __init__(self):
        super().__init__('mission_planner')

        self.detection_sub = self.create_subscription(
            String, '/safeguard/detections', self.on_detection, 10)

        self.panic_sub = self.create_subscription(
            String, '/safeguard/panic', self.on_panic, 10)

        self.command_pub = self.create_publisher(
            String, '/safeguard/commands', 10)

        self.get_logger().info('Mission Planner node started')

    def on_detection(self, msg):
        detections = json.loads(msg.data)
        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            priority = detection['priority']
            self.get_logger().info(
                f'Detection received: {label} ({confidence}%) priority={priority}')
            command = self.decide_action(detection)
            if command:
                cmd_msg = String()
                cmd_msg.data = json.dumps(command)
                self.command_pub.publish(cmd_msg)
                self.get_logger().info(f'Command published: {command["action"]}')

    def on_panic(self, msg):
        data = json.loads(msg.data)
        lat = data.get('latitude', 0.0)
        lng = data.get('longitude', 0.0)
        self.get_logger().info(f'PANIC received: {lat}, {lng}')
        command = {
            'action': 'dispatch_drone',
            'reason': 'PANIC_BUTTON',
            'confidence': 100,
            'latitude': lat,
            'longitude': lng,
            'priority': 'critical',
        }
        cmd_msg = String()
        cmd_msg.data = json.dumps(command)
        self.command_pub.publish(cmd_msg)
        self.get_logger().info('Drone dispatched to panic location')

    def decide_action(self, detection):
        label = detection['label']
        priority = detection['priority']
        if priority == 'high':
            return {
                'action': 'dispatch_drone',
                'reason': label,
                'confidence': detection['confidence'],
                'bbox': detection.get('bbox'),
            }
        elif priority == 'medium':
            return {
                'action': 'raise_alert',
                'reason': label,
                'confidence': detection['confidence'],
            }
        return None


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
