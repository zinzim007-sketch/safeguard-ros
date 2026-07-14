import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class MissionPlanner(Node):
    """
    SafeguardOS Mission Planner — Layer 6
    
    Subscribes to /safeguard/detections and decides autonomous actions:
    - PERSON detected in restricted zone → dispatch drone
    - VEHICLE after hours → raise alert
    - FIRE/SMOKE → immediate response
    
    Publishes commands to /safeguard/commands for the MAVLink bridge
    to execute on the drone.
    """

    def __init__(self):
        super().__init__('mission_planner')
        
        # Subscribe to detections from detector.py
        self.detection_sub = self.create_subscription(
            String,
            '/safeguard/detections',
            self.on_detection,
            10
        )
        
        # Publish commands to MAVLink bridge
        self.command_pub = self.create_publisher(
            String,
            '/safeguard/commands',
            10
        )
        
        self.get_logger().info('Mission Planner node started')

    def on_detection(self, msg):
        detections = json.loads(msg.data)
        
        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            priority = detection['priority']
            
            self.get_logger().info(
                f'Detection received: {label} ({confidence}%) priority={priority}'
            )
            
            # Decide action based on detection
            command = self.decide_action(detection)
            if command:
                cmd_msg = String()
                cmd_msg.data = json.dumps(command)
                self.command_pub.publish(cmd_msg)
                self.get_logger().info(f'Command published: {command["action"]}')

    def decide_action(self, detection):
        label = detection['label']
        priority = detection['priority']

        if priority == 'high':
            # Person or fire — dispatch drone to investigate
            return {
                'action': 'dispatch_drone',
                'reason': label,
                'confidence': detection['confidence'],
                'bbox': detection.get('bbox'),
            }
        
        elif priority == 'medium':
            # Vehicle — raise alert, operator decides
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
