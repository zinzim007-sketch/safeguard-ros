import os
import json
import time
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from pymavlink import mavutil


class MAVLinkBridge(Node):
    """
    SafeGuard OS MAVLink Bridge.

    ROS:
        /safeguard/commands

    Supported commands:
        dispatch_drone
        return_home
        raise_alert

    The bridge supports:
        - UDP MAVLink
        - serial MAVLink
        - dry-run mode for development

    Environment variables:

        MAVLINK_CONNECTION
            Example:
                udp:127.0.0.1:14540
                /dev/ttyUSB0
                /dev/ttyACM0

        MAVLINK_BAUD
            Serial baud rate.
            Default: 57600

        ENABLE_REAL_FLIGHT
            0 = dry run
            1 = allow real MAVLink flight commands

        TARGET_ALTITUDE
            Default dispatch altitude in metres.
    """

    def __init__(self):
        super().__init__('mavlink_bridge')

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------

        self.connection_string = os.getenv(
            'MAVLINK_CONNECTION',
            'udp:127.0.0.1:14540'
        )

        self.baud = int(
            os.getenv('MAVLINK_BAUD', '57600')
        )

        self.enable_real_flight = os.getenv(
            'ENABLE_REAL_FLIGHT',
            '0'
        ) == '1'

        self.target_altitude = float(
            os.getenv('TARGET_ALTITUDE', '20')
        )

        # ---------------------------------------------------------
        # MAVLink state
        # ---------------------------------------------------------

        self.master = None
        self.connected = False
        self.last_heartbeat = 0.0

        self.home_lat = None
        self.home_lon = None
        self.home_alt = None

        # Current patrol waypoint target.
        self.target_lat = None
        self.target_lon = None
        self.target_waypoint_index = None

        # Distance at which a patrol waypoint is considered reached.
        # This is deliberately conservative for the software test.
        self.waypoint_radius = float(
            os.getenv('WAYPOINT_RADIUS_METERS', '5')
        )

        # ---------------------------------------------------------
        # ROS subscriber
        # ---------------------------------------------------------

        self.command_sub = self.create_subscription(
            String,
            '/safeguard/commands',
            self.on_command,
            10
        )

        self.mission_event_pub = self.create_publisher(
            String,
            '/safeguard/mission_events',
            10
        )

        # Periodically check MAVLink connection/telemetry.
        self.timer = self.create_timer(
            1.0,
            self.check_connection
        )

        self.get_logger().info(
            '========================================'
        )
        self.get_logger().info(
            'SafeGuard MAVLink Bridge starting'
        )
        self.get_logger().info(
            f'Connection: {self.connection_string}'
        )
        self.get_logger().info(
            f'Real flight enabled: {self.enable_real_flight}'
        )
        self.get_logger().info(
            f'Dispatch altitude: {self.target_altitude} m'
        )
        self.get_logger().info(
            '========================================'
        )

        # Connect immediately, but do NOT crash ROS if PX4
        # is not connected yet.
        self.connect_mavlink()

    # =============================================================
    # MAVLink CONNECTION
    # =============================================================

    def connect_mavlink(self):
        """Connect to PX4 using pymavlink."""

        try:
            self.get_logger().info(
                f'Connecting to MAVLink: {self.connection_string}'
            )

            if self.connection_string.startswith('udp:'):
                self.master = mavutil.mavlink_connection(
                    self.connection_string
                )
            else:
                self.master = mavutil.mavlink_connection(
                    self.connection_string,
                    baud=self.baud
                )

            # Wait briefly for a heartbeat.
            self.get_logger().info(
                'Waiting for PX4 heartbeat...'
            )

            heartbeat = self.master.wait_heartbeat(
                timeout=3
            )

            if heartbeat is None:
                self.get_logger().warning(
                    'No PX4 heartbeat received. '
                    'Bridge will remain running.'
                )
                self.connected = False
                return

            self.connected = True
            self.last_heartbeat = time.time()

            self.get_logger().info(
                'PX4 HEARTBEAT RECEIVED'
            )

            self.get_logger().info(
                f'System ID: {self.master.target_system}'
            )

            self.get_logger().info(
                f'Component ID: {self.master.target_component}'
            )

            # Request GPS position messages.
            self.request_position_stream()

        except Exception as e:
            self.connected = False
            self.master = None

            self.get_logger().warning(
                f'MAVLink connection unavailable: {e}'
            )

            self.get_logger().warning(
                'ROS bridge will continue running.'
            )

    def distance_meters(
        self,
        lat1,
        lon1,
        lat2,
        lon2
    ):
        """
        Approximate horizontal distance between two GPS coordinates.

        Returns distance in metres.
        """

        earth_radius = 6371000.0

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            +
            math.cos(lat1_rad)
            * math.cos(lat2_rad)
            * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )

        return earth_radius * c

    def check_connection(self):
        """Read MAVLink messages without blocking ROS."""

        if self.master is None:
            return

        try:
            while True:
                msg = self.master.recv_match(
                    blocking=False
                )

                if msg is None:
                    break

                msg_type = msg.get_type()

                if msg_type == 'HEARTBEAT':
                    self.connected = True
                    self.last_heartbeat = time.time()


                elif msg_type == 'GLOBAL_POSITION_INT':

                    current_lat = msg.lat / 1e7
                    current_lon = msg.lon / 1e7
                    current_alt = msg.relative_alt / 1000.0

                    self.home_lat = current_lat
                    self.home_lon = current_lon
                    self.home_alt = current_alt

                    # ---------------------------------------------------------
                    # PATROL WAYPOINT ARRIVAL
                    # ---------------------------------------------------------

                    if (
                        self.target_lat is not None
                        and self.target_lon is not None
                        and self.target_waypoint_index is not None
                    ):

                        distance = self.distance_meters(
                            current_lat,
                            current_lon,
                            self.target_lat,
                            self.target_lon
                        )

                        self.get_logger().debug(
                            f'Waypoint distance: {distance:.1f} m'
                        )

                        if distance <= self.waypoint_radius:

                            waypoint_index = (
                                self.target_waypoint_index
                            )

                            self.get_logger().info(
                                f'Waypoint {waypoint_index} reached '
                                f'({distance:.1f} m from target).'
                            )

                            self.publish_mission_event({
                                'event': 'waypoint_reached',
                                'waypoint_index': waypoint_index,
                                'latitude': current_lat,
                                'longitude': current_lon,
                                'distance_meters': distance,
                            })

                            # Clear target so we don't publish the same
                            # waypoint_reached event repeatedly.
                            self.target_lat = None
                            self.target_lon = None
                            self.target_waypoint_index = None



        except Exception as e:
            self.get_logger().warning(
                f'MAVLink receive error: {e}'
            )

    def request_position_stream(self):
        """Ask PX4 for regular global position updates."""

        if not self.master:
            return

        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                500000,   # 2 Hz
                0,
                0,
                0,
                0,
                0
            )

        except Exception as e:
            self.get_logger().warning(
                f'Could not request position stream: {e}'
            )

    # =============================================================
    # ROS COMMAND HANDLER
    # =============================================================

    def on_command(self, msg):
        """Receive commands from Mission Planner."""

        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(
                f'Invalid command JSON: {msg.data}'
            )
            return

        action = command.get('action')

        self.get_logger().info(
            f'Command received: {action}'
        )

        if action == 'dispatch_drone':
            self.dispatch_drone(command)

        elif action == 'return_home':
            self.return_home()

        elif action == 'raise_alert':
            self.raise_alert(command)

        else:
            self.get_logger().warning(
                f'Unknown command: {action}'
            )

    # =============================================================
    # DISPATCH
    # =============================================================

    def dispatch_drone(self, command):
        """
        Dispatch ARGUS to a GPS location.

        Expected command:

        {
            "action": "dispatch_drone",
            "latitude": -33.93,
            "longitude": 18.86,
            "reason": "PANIC_BUTTON"
        }
        """

        latitude = command.get('latitude')
        longitude = command.get('longitude')

        reason = command.get(
            'reason',
            'UNKNOWN'
        )

        if latitude is None or longitude is None:
            self.get_logger().error(
                'Dispatch rejected: latitude/longitude missing.'
            )
            return

        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError):
            self.get_logger().error(
                'Dispatch rejected: invalid GPS coordinates.'
            )
            return

        # Basic coordinate sanity check.
        if not (-90 <= latitude <= 90):
            self.get_logger().error(
                f'Invalid latitude: {latitude}'
            )
            return

        if not (-180 <= longitude <= 180):
            self.get_logger().error(
                f'Invalid longitude: {longitude}'
            )
            return

        self.get_logger().info(
            '----------------------------------------'
        )

        self.get_logger().info(
            f'DISPATCH REQUEST'
        )

        self.get_logger().info(
            f'Reason: {reason}'
        )

        self.get_logger().info(
            f'Target: {latitude}, {longitude}'
        )

        self.get_logger().info(
            f'Altitude: {self.target_altitude} m'
        )

        self.get_logger().info(
            '----------------------------------------'
        )

        if reason == 'PATROL_WAYPOINT':

            self.target_lat = latitude
            self.target_lon = longitude

            self.target_waypoint_index = command.get(
                'waypoint_index'
            )

            self.get_logger().info(
                f'Patrol target registered: '
                f'waypoint {self.target_waypoint_index}'
            )

        # ---------------------------------------------------------
        # DEVELOPMENT SAFETY


        # ---------------------------------------------------------


        if not self.enable_real_flight:

            self.get_logger().warning(
                'DRY RUN: real flight commands are DISABLED.'
            )

            self.get_logger().info(
                f'Would dispatch ARGUS to '
                f'{latitude}, {longitude}'
            )

            return



        # ---------------------------------------------------------
        # REAL MAVLINK COMMAND
        # ---------------------------------------------------------

        if self.master is None or not self.connected:
            self.get_logger().error(
                'Cannot dispatch: PX4 is not connected.'
            )
            return

        try:
            self.send_reposition(
                latitude,
                longitude,
                self.target_altitude
            )

            self.get_logger().info(
                'ARGUS dispatch command sent to PX4.'
            )

        except Exception as e:
            self.get_logger().error(
                f'Dispatch failed: {e}'
            )

    # =============================================================
    # MAVLINK REPOSITION
    # =============================================================

    def send_reposition(
        self,
        latitude,
        longitude,
        altitude
    ):
        """
        Send MAV_CMD_DO_REPOSITION to PX4.

        PX4 expects:
            param1 = ground speed
            param2 = bitmask
            param3 = loiter radius
            param4 = yaw
            param5 = latitude
            param6 = longitude
            param7 = altitude

        IMPORTANT:
        The altitude supplied here is interpreted as the altitude
        expected by PX4 for this command. We will verify the correct
        altitude reference on the actual aircraft before enabling
        real flight.
        """

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,

            mavutil.mavlink.MAV_CMD_DO_REPOSITION,

            0,              # confirmation

            -1,             # param1: default ground speed
            0,              # param2: bitmask
            0,              # param3: loiter radius
            float('nan'),   # param4: yaw unchanged

            latitude,       # param5: latitude
            longitude,      # param6: longitude
            altitude        # param7: altitude
        )



    # =============================================================
    # RETURN TO LAUNCH
    # =============================================================

    def return_home(self):
        """Send RTL command to PX4."""

        self.get_logger().info(
            'RETURN TO HOME requested.'
        )

        if not self.enable_real_flight:
            self.get_logger().warning(
                'DRY RUN: RTL command NOT sent.'
            )
            return

        if self.master is None or not self.connected:
            self.get_logger().error(
                'Cannot RTL: PX4 is not connected.'
            )
            return

        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,

                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,

                0,

                0,
                0,
                0,
                0,
                0,
                0,
                0
            )

            self.get_logger().info(
                'RTL command sent to PX4.'
            )

        except Exception as e:
            self.get_logger().error(
                f'RTL failed: {e}'
            )

    # =============================================================
    # ALERT
    # =============================================================

    def raise_alert(self, command):
        """Handle a detection alert."""

        reason = command.get(
            'reason',
            'Unknown'
        )

        confidence = command.get(
            'confidence',
            0
        )

        self.get_logger().warning(
            f'ALERT: {reason} '
            f'({confidence}% confidence)'
        )


    # =============================================================
    # MISSION EVENTS
    # =============================================================

    def publish_mission_event(self, event):
        """Publish an event back to the Mission Planner."""

        msg = String()
        msg.data = json.dumps(event)

        self.mission_event_pub.publish(msg)

        self.get_logger().info(
            f'Mission event published: {event.get("event")}'
        )

        # Alerts are handled by the Flutter application.
        # No flight command is sent here.

    # =============================================================
    # CLEANUP
    # =============================================================

    def destroy_node(self):
        """Close MAVLink connection cleanly."""

        try:
            if self.master:
                self.master.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MAVLinkBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()