import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.action import CancelResponse
import time
from bronchoscopy_interfaces.action import Arm
from bronchoscopy_interfaces.srv import TrackContour
from bronchoscopy_interfaces.srv import SensePressure
from bronchoscopy_interfaces.msg import DaqOutput
from bronchoscopy_interfaces.srv import MoveUR
import rtde_rotation
import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class ArmAction(Node):
    def __init__(self):
        super().__init__('armaction')
        self._action_server = ActionServer(self, Arm, 'arm', execute_callback=self.movearm, callback_group=None, cancel_callback=self.cancel_callback)
        # publisher for sending commands to the pressure regulator
        #self.ur_pub = self.create_publisher(Twist, 'UR5e_motion', 10)

        # service clients for obtaining necessary data
        contour_cb_group = ReentrantCallbackGroup()
        self.cli = self.create_client(TrackContour, 'track_contour', callback_group=contour_cb_group)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        #self.pressure_cli = self.create_client(SensePressure, 'sense_pressure')
        #while not self.pressure_cli.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info('service not available, waiting again...')
        self.pres_req = SensePressure.Request()
        self.subscriber = self.create_subscription(DaqOutput, 'pressure', self.collect_pressure, 10, callback_group=ReentrantCallbackGroup())
        self.pressure = 0

        #self.RobArm = rtde_rotation.RobotArm()
        self.get_logger().info("Robot Initialized")
        #self.init_pose = self.RobArm.rtde_r.getActualTCPPose()
        self.total_translation_x = 0
        self.total_translation_z = 0
        self.tracked_rotation = 0

        self.ur_cli = self.create_client(MoveUR, 'move_UR')
        while not self.ur_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('UR service not available, waiting again...')

    def cancel_callback(self, goal_handle):
        # Accepts or rejects a client request to cancel an action
        self.get_logger().info('Received cancel request :(')
        return CancelResponse.ACCEPT

    async def send_ur_request(self, rot, t_x, t_z):
        self.ur_req.rotation = rot
        self.ur_req.translate_x = t_x
        self.ur_req.translate_z = t_z
        self.ur_future = self.ur_cli.call_async(self.ur_req)
        await self.ur_future
        #rclpy.spin_until_future_complete(self, self.ur_future)
        return self.ur_future.result()

    async def send_request(self, a, b, c):
        self.req.n_path_points = a
        self.req.bend_direction = b
        self.req.path_projection = c
        self.future = self.cli.call_async(self.req)
        await self.future
        #rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    async def movearm(self, goal_handle):
        nPathPoints = 10
        target_bend_angle = goal_handle.request.path_angle
        target_rotation_angle = goal_handle.request.rotation_angle
        distances = goal_handle.request.displacement
        path_proj = goal_handle.request.path_projection
        bend_dir = goal_handle.request.bend_direction
        #press_resp = self.send_pressure_request(True)
        #pressure = press_resp.pressure
        feedback_msg = Arm.Feedback()
        self.req = TrackContour.Request()
        self.ur_req = MoveUR.Request()

        total_rotation = self.tracked_rotation
        #bend_dir = np.array([-0.97422725, 0.22556875])  # obtained from steering_direction_test.py
        MAX_ROT = 2.9  #1.83
        MIN_ROT = -1.68
        if target_rotation_angle != 0:
            #resp_pose = await self.send_ur_request(target_rotation_angle, 0, 0)
            #time.sleep(1)
            #self.pressure = 0
            if target_bend_angle < 0.0 and self.pressure >= 100:
                bend_dir = -bend_dir
                response = await self.send_request(nPathPoints, bend_dir, path_proj)
                rotation_feedback = response.rotation
            else:
                bend_dir = bend_dir
                response = await self.send_request(nPathPoints, bend_dir, path_proj)
                rotation_feedback = response.rotation
            self.get_logger().info('Running' + str(rotation_feedback))
            feedback_msg.rotation_angle = rotation_feedback
            goal_handle.publish_feedback(feedback_msg)
            while abs(rotation_feedback) > 0.2:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self.get_logger().info('Goal canceled')
                    return Arm.Result()
                # send through controller then into msg
                rotation_cmd = -rotation_feedback
                self.get_logger().info('Rotation: ' + str(rotation_cmd))
                #ur5e_msg.angular.y = rotation_cmd
                #self.ur_pub.publish(ur5e_msg)

                #if self.tracked_rotation + rotation_cmd < MIN_ROT and self.tracked_rotation + (np.pi-rotation_cmd) < MAX_ROT:
                #    rotation_cmd = np.pi - rotation_cmd

                total_rotation += rotation_cmd

                # Safety condition maintaining rotation within suitable range
                if total_rotation > MAX_ROT and self.tracked_rotation < MAX_ROT-0.1:
                    rotation_cmd = 0.1
                elif total_rotation < MIN_ROT and self.tracked_rotation > MIN_ROT+0.1:
                    rotation_cmd = -0.1
                if self.tracked_rotation >= MAX_ROT-0.1 or self.tracked_rotation <= MIN_ROT+0.1:
                    break
                # Apply rotation to UR5e
                #new_pose = self.RobArm.rotate_abt_axis(rotation_cmd, 4, 0.1)
                resp_pose = await self.send_ur_request(rotation_cmd, 0, 0)
                self.tracked_rotation += rotation_cmd
                time.sleep(1)
                # Obtain contour location feedback from camera
                response = await self.send_request(nPathPoints, bend_dir, path_proj)
                if response.rotation == 0 and response.distance_x == 999:
                    rotation_cmd = target_rotation_angle - total_rotation
                    resp_pose = await self.send_ur_request(rotation_cmd, 0, 0)
                    rotation_feedback = 0
                elif np.sqrt(response.distance_x**2+response.distance_y**2) < 30:
                    rotation_cmd = 0
                    rotation_feedback = 0.02
                else:
                    rotation_feedback = response.rotation
                feedback_msg.rotation_angle = -rotation_feedback
                # if tracked contour was lost before finishing rotation, action is canceled
                if rotation_feedback == 0:
                    goal_handle.canceled()
                    return Arm.result()
                goal_handle.publish_feedback(feedback_msg)
        # Reset UR5e orientation after bending
        elif distances[0] == 0 and distances[1] == 0:
            resp_pose = await self.send_ur_request(-self.tracked_rotation, 0, 0)
            self.tracked_rotation = 0
        else:
            # publish command to UR5e
            if abs(distances[0]) < 5 and abs(distances[1]) < 5:
                # Safety condition
                if abs(self.total_translation_x) > 15 or abs(self.total_translation_z) > 15:
                    goal_handle.succeed()
                    result = Arm.Result()
                    return result
                self.total_translation_x += distances[0]
                self.total_translation_z += distances[1]
                dist_x = 0.001*distances[0]  # converting to mm
                dist_z = 0.001*distances[1]
                #new_pose = self.RobArm.translate_axis([dist_x, dist_z], [0, 2])
                resp_pose = await self.send_ur_request(0, dist_x, dist_z)

        goal_handle.succeed()
        result = Arm.Result()
        result.total_rotation = self.tracked_rotation

        return result

    def collect_pressure(self, msg):
        self.pressure = msg.pressure
        self.tip_pos_bframe = msg.tip_position_baseframe
        self.tip_ori_bframe = msg.tip_orientation_baseframe
        self.bend_angle = msg.bend_angle


def main(args=None):
    rclpy.init(args=args)
    armaction = ArmAction()
    executor = MultiThreadedExecutor()
    executor.add_node(armaction)
    executor.spin()
    armaction.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
