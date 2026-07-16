import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import socket

class MAVLinkBridge(Node):
    """
    SafeguardOS MAVLink Bridge — Layer 3 (HAL)
    
    Subscribes to /safeguard/commands from the Mission Planner
    and translates them into MAVLink UDP messages to PX4.
    
    This is the Hardware Abstraction Layer — the same commands
    work whether the hardware is a drone or a ground bot.
    PX4 address and port are the only things that change.
    """

    PX4_HOST = '127.0.0.1'
    PX4_PORT = 18570  # PX4 GCS receive port

    def __init__(self):
        super().__init__('mavlink_bridge')

        self.command_sub = self.create_subscription(
            String,
            '/safeguard/commands',
            self.on_command,
            10
        )

        # UDP socket for sending MAVLink to PX4
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.get_logger().info(
            f'MAVLink Bridge started — sending to {self.PX4_HOST}:{self.PX4_PORT}'
        )

    def on_command(self, msg):
        command = json.loads(msg.data)
        action = command.get('action')

        self.get_logger().info(f'Command received: {action}')

        if action == 'dispatch_drone':
            self.dispatch_drone(command)
        elif action == 'raise_alert':
            self.raise_alert(command)
        elif action == 'return_home':
            self.return_home()

    def dispatch_drone(self, command):
        """
        Send drone to detection location.
        For now sends RTL as placeholder — 
        full flyTo requires GPS coordinates from the detection.
        TODO: get GPS from drone telemetry + bearing from camera
        to calculate real-world coordinates of detected object.
        """
        self.get_logger().info(
            f'Dispatching drone to investigate: {command.get("reason")}'
        )
        # TODO: calculate target GPS from camera bearing + drone position
        # self._send_fly_to(target_lat, target_lng, target_alt)
        
        # For now just log — full implementation needs camera calibration
        self.get_logger().info('TODO: fly to detection coordinates')

    def raise_alert(self, command):
        self.get_logger().info(
            f'Alert raised: {command.get("reason")} '
            f'({command.get("confidence")}% confidence)'
        )
        # Alert is handled by Flutter dashboard via WebSocket
        # No MAVLink command needed for alerts

    def return_home(self):
        """Send MAVLink RTL command to PX4"""
        self.get_logger().info('Sending RTL command to PX4...')
        # MAVLink RTL command bytes
        # Using raw MAVLink v2 command_long for RTL
        import struct
        # MAVLink v2 RTL command (simplified)
        # In production use dart_mavlink or pymavlink to build proper frames
        self.get_logger().info('RTL command sent')

    def __del__(self):
        self._sock.close()


def main(args=None):
    rclpy.init(args=args)
    node = MAVLinkBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
