import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from bronchoscopy_interfaces.msg import DaqInput
from std_msgs.msg import Bool

class JoyListener(Node):
    def __init__(self):
        super().__init__('joy_listener')
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)
        self.publisher = self.create_publisher(Twist, 'UR5e_motion', 10)
        self.publisher2 = self.create_publisher(DaqInput, 'daq_input', 10)
        self.teleop_pub = self.create_publisher(Bool, 'teleoperation', 10)
        self.lungmodel = self.create_publisher(Bool, 'cameracontrol', 10)
        self.activated = True  # Robot control
        self.activated2 = False  # Simulation camera control
        self.press_count = 0
        self.press_count2 = 0
        self.button_pressed = False  # Track if the button was pressed
        self.button_pressed2 = False
        self.subscription  # prevent unused variable warning

    def joy_callback(self, msg):

        # control lung simulation camera
        if msg.buttons[4] == 1:
            if not self.button_pressed2:
                self.button_pressed2 = True
                self.press_count2 += 1
                self.activated2 = (self.press_count2 % 2) == 1
                state2 = 'activated' if self.activated2 else 'deactivated'
                self.get_logger().info(f'Toggle button pressed. Lung model camera {state2}.')
                t_msg2 = Bool()
                t_msg2.data = self.activated2
                self.lungmodel.publish(t_msg2)
        else:
            self.button_pressed2 = False

        # control robot arm when 'start' button toggle is pressed
        if msg.buttons[6] == 1:
            # Ensure toggle only happens once per button press
            if not self.button_pressed:
                self.button_pressed = True
                self.press_count += 1
                self.activated = (self.press_count % 2) == 0  # True for teleoperation
                state = 'activated' if self.activated else 'deactivated'
                self.get_logger().info(f'Toggle button pressed. Teleoperation {state}.')
                t_msg = Bool()
                t_msg.data = self.activated
                self.teleop_pub.publish(t_msg)
        else:
            self.button_pressed = False

        # Control robot only when not controlling lung simulation camera
        if self.activated and not self.activated2:
            twist = Twist()
            daq_in = DaqInput()

            #initialize
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            daq_in.steps = 0
            daq_in.pressure = 0

            #map d-pad for linear motion
            if msg.buttons[12] == 1:  #up
                twist.linear.z = 1.0
            if msg.buttons[11] == 1:  #down
                twist.linear.z = -1.0
            if msg.buttons[13] == 1:  #left
                twist.linear.x = 1.0
            if msg.buttons[14] == 1:  #right
                twist.linear.x = -1.0

            #map x and y for insertion and retraction
            if msg.buttons[0] == 1:  #retract
                daq_in.steps = -1
            elif msg.buttons[3] == 1:  #insert
                daq_in.steps = 1

            #map rotation to right joystick
            twist.angular.y = -0.2 * msg.axes[2]

            #map pressure to right trigger
            max_pressure = 250
            daq_in.pressure = -(max_pressure/2)*msg.axes[5] + (max_pressure/2)

            daq_in.contour_distance = [0, 0]

            self.publisher.publish(twist)
            self.get_logger().info('Publish Twist: linear_z="%f", linear_x="%f"' % (twist.linear.z, twist.linear.x))
            self.publisher2.publish(daq_in)
            self.get_logger().info('Publish Twist: angular_x="%f"' % (twist.angular.x))
            self.get_logger().info('Publish Twist: steps="%f"' % (daq_in.steps))

            self.get_logger().info('Axes: "%s"' % str(msg.axes))
            #self.get_logger().info('Buttons: "%s"' % str(msg.buttons))

def main(args=None):
    rclpy.init(args=args)
    node = JoyListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
