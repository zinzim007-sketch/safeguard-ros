import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MissionPlanner(Node):

    def __init__(self):
        super().__init__('mission_planner')

        # =========================================================
        # MISSION STATE
        # =========================================================

        self.state = 'IDLE'

        self.waypoints = []
        self.current_waypoint = 0

        # =========================================================
        # ROS SUBSCRIPTIONS
        # =========================================================

        self.detection_sub = self.create_subscription(
            String,
            '/safeguard/detections',
            self.on_detection,
            10
        )

        self.panic_sub = self.create_subscription(
            String,
            '/safeguard/panic',
            self.on_panic,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/safeguard/mission_control',
            self.on_mission_control,
            10
        )

        self.mission_event_sub = self.create_subscription(
            String,
            '/safeguard/mission_events',
            self.on_mission_event,
            10
        )

        # =========================================================
        # ROS PUBLISHERS
        # =========================================================

        self.command_pub = self.create_publisher(
            String,
            '/safeguard/commands',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/safeguard/mission_status',
            10
        )

        self.get_logger().info(
            'Mission Planner node started'
        )

        self.publish_status()

    # =============================================================
    # STATE MANAGEMENT
    # =============================================================

    def set_state(self, new_state):
        """Change mission state and notify the operator."""

        old_state = self.state
        self.state = new_state

        self.get_logger().info(
            f'Mission state: {old_state} -> {new_state}'
        )

        self.publish_status()

    # =============================================================
    # STATUS
    # =============================================================

    def publish_status(self):
        """Publish current mission status."""

        status = {
            'state': self.state,
            'current_waypoint': self.current_waypoint,
            'total_waypoints': len(self.waypoints),
        }

        msg = String()
        msg.data = json.dumps(status)

        self.status_pub.publish(msg)

    # =============================================================
    # MISSION CONTROL
    # =============================================================

    def on_mission_control(self, msg):
        """
        Receive operator mission commands.

        Supported:

            start
            pause
            resume
            return_home
            abort
        """

        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(
                f'Invalid mission control JSON: {msg.data}'
            )
            return

        action = command.get('action')

        self.get_logger().info(
            f'Mission control received: {action}'
        )

        # ---------------------------------------------------------
        # START
        # ---------------------------------------------------------

        if action == 'start':

            if self.state not in ['IDLE', 'COMPLETE']:
                self.get_logger().warning(
                    f'Cannot start mission while state is '
                    f'{self.state}'
                )
                return

            self.waypoints = command.get(
                'waypoints',
                []
            )

            self.current_waypoint = 0

            if not self.waypoints:
                self.get_logger().warning(
                    'Cannot start patrol: no waypoints supplied.'
                )
                return

            self.set_state('PATROLLING')

            self.send_next_waypoint()

        # ---------------------------------------------------------
        # PAUSE
        # ---------------------------------------------------------

        elif action == 'pause':

            if self.state != 'PATROLLING':
                self.get_logger().warning(
                    f'Cannot pause while state is {self.state}'
                )
                return

            self.set_state('PAUSED')

            self.publish_command({
                'action': 'pause_mission',
                'reason': 'OPERATOR_REQUEST'
            })

        # ---------------------------------------------------------
        # RESUME
        # ---------------------------------------------------------

        elif action == 'resume':

            if self.state != 'PAUSED':
                self.get_logger().warning(
                    f'Cannot resume while state is {self.state}'
                )
                return

            self.set_state('PATROLLING')

            self.send_next_waypoint()

        # ---------------------------------------------------------
        # RETURN HOME
        # ---------------------------------------------------------

        elif action == 'return_home':

            if self.state == 'IDLE':
                self.get_logger().warning(
                    'Drone is already idle.'
                )
                return

            self.set_state('RETURNING_HOME')

            self.publish_command({
                'action': 'return_home',
                'reason': 'OPERATOR_REQUEST'
            })

        # ---------------------------------------------------------
        # ABORT
        # ---------------------------------------------------------

        elif action == 'abort':

            self.set_state('RETURNING_HOME')

            self.publish_command({
                'action': 'return_home',
                'reason': 'MISSION_ABORT'
            })

        else:

            self.get_logger().warning(
                f'Unknown mission control action: {action}'
            )

    # =============================================================
    # WAYPOINT MANAGEMENT
    # =============================================================

    def send_next_waypoint(self):
        """Send the current waypoint to the MAVLink layer."""

        if self.state != 'PATROLLING':
            return

        if self.current_waypoint >= len(self.waypoints):

            self.set_state('COMPLETE')

            self.get_logger().info(
                'Patrol complete.'
            )

            return

        waypoint = self.waypoints[
            self.current_waypoint
        ]

        latitude = waypoint.get('latitude')
        longitude = waypoint.get('longitude')

        if latitude is None or longitude is None:

            self.get_logger().error(
                f'Invalid waypoint: {waypoint}'
            )

            self.set_state('RETURNING_HOME')

            self.publish_command({
                'action': 'return_home',
                'reason': 'INVALID_WAYPOINT'
            })

            return

        command = {
            'action': 'dispatch_drone',
            'reason': 'PATROL_WAYPOINT',
            'latitude': latitude,
            'longitude': longitude,
            'waypoint_index': self.current_waypoint,
            'total_waypoints': len(self.waypoints),
            'priority': 'normal',
        }

        self.publish_command(command)

        self.get_logger().info(
            f'Sent waypoint '
            f'{self.current_waypoint + 1}/'
            f'{len(self.waypoints)}'
        )

    def waypoint_reached(self):
        """
        Called when the flight system reports that
        the current waypoint has been reached.
        """

        if self.state != 'PATROLLING':
            return

        self.get_logger().info(
            f'Waypoint {self.current_waypoint + 1} reached.'
        )

        self.current_waypoint += 1

        self.publish_status()

        if self.current_waypoint >= len(self.waypoints):

            self.set_state('COMPLETE')

            self.get_logger().info(
                'All patrol waypoints completed.'
            )

        else:

            self.send_next_waypoint()

    # =============================================================
    # PANIC
    # =============================================================

    def on_panic(self, msg):
        """
        Handle panic GPS coordinates.

        Panic always overrides the current patrol.
        """

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:

            self.get_logger().error(
                f'Invalid panic JSON: {msg.data}'
            )

            return

        latitude = data.get('latitude')
        longitude = data.get('longitude')

        if latitude is None or longitude is None:

            self.get_logger().error(
                'PANIC rejected: GPS coordinates missing.'
            )

            return

        self.get_logger().warning(
            f'PANIC received: '
            f'{latitude}, {longitude}'
        )

        # Panic overrides whatever mission is currently running.
        self.set_state('EMERGENCY')

        command = {
            'action': 'dispatch_drone',
            'reason': 'PANIC_BUTTON',
            'confidence': 100,
            'latitude': latitude,
            'longitude': longitude,
            'priority': 'critical',
        }

        self.publish_command(command)

        self.get_logger().warning(
            'ARGUS dispatched to panic location.'
        )


    # =============================================================
    # MISSION EVENTS
    # =============================================================

    def on_mission_event(self, msg):
        """
        Receive events from the flight-control layer.

        Currently supported:

            waypoint_reached
        """

        try:
            event = json.loads(msg.data)

        except json.JSONDecodeError:
            self.get_logger().error(
                f'Invalid mission event JSON: {msg.data}'
            )
            return

        event_type = event.get('event')

        if event_type == 'waypoint_reached':

            waypoint_index = event.get(
                'waypoint_index'
            )

            self.get_logger().info(
                f'Waypoint reached: {waypoint_index}'
            )

            # Make sure the event refers to the waypoint
            # we are currently waiting for.
            if waypoint_index != self.current_waypoint:
                self.get_logger().warning(
                    f'Ignoring waypoint event for '
                    f'{waypoint_index}; currently waiting for '
                    f'{self.current_waypoint}'
                )
                return

            self.waypoint_reached()

        else:

            self.get_logger().warning(
                f'Unknown mission event: {event_type}'
            )

    # =============================================================
    # DETECTIONS
    # =============================================================

    def on_detection(self, msg):

        try:
            detections = json.loads(msg.data)

        except json.JSONDecodeError:

            self.get_logger().error(
                f'Invalid detection JSON: {msg.data}'
            )

            return

        for detection in detections:

            label = detection.get(
                'label',
                'unknown'
            )

            confidence = detection.get(
                'confidence',
                0
            )

            priority = detection.get(
                'priority',
                'low'
            )

            self.get_logger().info(
                f'Detection received: '
                f'{label} '
                f'({confidence}%) '
                f'priority={priority}'
            )

            command = self.decide_action(
                detection
            )

            if command:

                self.publish_command(
                    command
                )

    # =============================================================
    # DETECTION DECISION LOGIC
    # =============================================================

    def decide_action(self, detection):
        label = detection['label']
        priority = detection['priority']

        # Loitering is a behavioural warning.
        # Do not immediately dispatch a drone just because
        # the detector marked it as critical.
        if label == 'LOITERING':
            return {
                'action': 'raise_alert',
                'reason': 'LOITERING',
                'confidence': detection['confidence'],
                'bbox': detection.get('bbox'),
                'priority': priority,
            }

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



    # =============================================================
    # COMMAND PUBLISHING
    # =============================================================

    def publish_command(self, command):

        msg = String()

        msg.data = json.dumps(
            command
        )

        self.command_pub.publish(
            msg
        )

        self.get_logger().info(
            f'Command published: '
            f'{command.get("action")}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = MissionPlanner()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()