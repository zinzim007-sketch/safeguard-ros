import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('detection_publisher')
        self.publisher_ = self.create_publisher(String, '/safeguard/detections', 10)
        self.get_logger().info('Detection publisher node started')

    def publish_detection(self, detection: dict):
        msg = String()
        msg.data = json.dumps(detection)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {detection["label"]} {detection["confidence"]}%')

def main(args=None):
    rclpy.init(args=args)
    node = DetectionPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
