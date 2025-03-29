import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import rtde_rotation
from bronchoscopy_interfaces.srv import MoveUR

class ArmSub(Node):
    def __init__(self):
        super().__init__('ur5esub')
        self.subscriber = self.create_subscription(Twist, 'UR5e_motion', self.motion_callback, 10)
        self.srv = self.create_service(MoveUR, 'move_UR', self.motion_service)
        self.RobArm = rtde_rotation.RobotArm()
        self.get_logger().info("Robot Initialized")

    def motion_callback(self, msg):
        self.get_logger().info('Received: ' + str(msg.linear.x) + 'm, ' + str(msg.linear.z) + 'm, ' + str(msg.angular.y) + 'rad')
        if msg.linear.x != 0 or msg.linear.z != 0:
            axes_trans = [0, 2]
            trans_dists = [0.001*msg.linear.x, 0.001*msg.linear.z]
            new_pose = self.RobArm.translate_axis(trans_dists, axes_trans)

        if msg.angular.y != 0:
            rot_axis = 4
            angle = msg.angular.y
            #velocity = msg.angular.x
            new_pose = self.RobArm.rotate_abt_axis(angle, rot_axis)

    def motion_service(self, request, response):
        self.get_logger().info('srv: ' + str(request.rotation) + str(request.translate_x))
        angle = request.rotation
        rot_axis = 4
        axes_trans = [0, 2]
        trans_dists = [request.translate_x, request.translate_z]
        if angle != 0:
            new_pose = self.RobArm.rotate_abt_axis(angle, rot_axis)
        else:
            new_pose = self.RobArm.translate_axis(trans_dists, axes_trans)
        response.new_pose = new_pose
        return response


def main(args=None):
    rclpy.init(args=args)
    ur5esub = ArmSub()
    rclpy.spin(ur5esub)
    ur5esub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
