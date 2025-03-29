import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.action import CancelResponse
import time
from bronchoscopy_interfaces.action import Steer
from bronchoscopy_interfaces.srv import TrackContour
import numpy as np
from bronchoscopy_interfaces.srv import SensePressure
from bronchoscopy_interfaces.msg import DaqInput


class SteeringAction(Node):
    def __init__(self):
        super().__init__('steeringaction')
        self._action_server = ActionServer(self, Steer, 'steer', execute_callback=self.bending, cancel_callback=self.cancel_callback)
        # publisher for sending commands to the pressure regulator
        self.daq_pub = self.create_publisher(DaqInput, 'daq_input', 10)

        # service clients for obtaining pressure data
        self.cli = self.create_client(TrackContour, 'track_contour')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = TrackContour.Request()
        self.pressure_cli = self.create_client(SensePressure, 'sense_pressure')
        while not self.pressure_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.pressure_req = SensePressure.Request()

    def cancel_callback(self, goal_handle):
        # Accepts or rejects a client request to cancel an action
        self.get_logger().info('Received cancel request :(')
        return CancelResponse.ACCEPT

    async def send_pressure_request(self, a):
        self.pressure_req.steering = a
        self.p_future = self.pressure_cli.call_async(self.pressure_req)
        await self.p_future
        #rclpy.spin_until_future_complete(self, self.p_future)
        return self.p_future.result()

    async def send_request(self, a, b, c):
        self.req.n_path_points = a
        self.req.bend_direction = b
        self.req.path_projection = c
        #self.req.bend_direction[1] = b[1]
        self.future = self.cli.call_async(self.req)
        await self.future
        #rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    async def bending(self, goal_handle):
        # Insert robot, wait for finish, and return when finished
        n_steps = goal_handle.request.n_steps
        if n_steps != 0:
            self.get_logger().info('Inserting...')
            stepper_msg = DaqInput()
            stepper_msg.steps = n_steps
            stepper_msg.contour_distance = [999, 999]
            self.daq_pub.publish(stepper_msg)
            time.sleep(n_steps*0.2)
            goal_handle.succeed()
            result = Steer.Result()
            result.steps = n_steps
            return result

        NPoints = 20
        self.get_logger().info("Bending...")
        p_response = await self.send_pressure_request(True)
        current_pressure = p_response.pressure
        tip_pos_bframe = p_response.tip_position_baseframe
        tip_ori_bframe = p_response.tip_orientation_baseframe
        bend_angle = p_response.bend_angle

        feedback_msg = Steer.Feedback()
        feedback_msg.angle_feedback = bend_angle
        # call service to check for tracked contour location
        path = goal_handle.request.path_projection
        start_press = current_pressure
        path_angle_error = goal_handle.request.path_angle
        # Release bending 
        if start_press > 100 and path_angle_error == 0:
            pressure_msg = DaqInput()
            pressure_msg.pressure = 100.0
            pressure_msg.contour_distance = [0, 0]
            self.daq_pub.publish(pressure_msg)
            goal_handle.succeed()
            result = Steer.Result()
            result.tip_pose.x = 0.0
            result.tip_pose.y = 0.0
            result.tip_pose.z = 0.0
            return result

        angle_target = bend_angle + (-path_angle_error)
        self.get_logger().info('bend: ' + str(bend_angle) + ' path: ' + str(path_angle_error))
        angle_target_deg = angle_target * (180/np.pi)
        press_input = (-0.1218*angle_target_deg**2 + 10.6439*abs(angle_target_deg) + 49.2694)/2  #4/24 -0.0785*angle_target_deg**2 + 7.7627*angle_target_deg + 102.1944
        bend_dir = goal_handle.request.bend_direction  #[-0.97422725, 0.22556875]  # obtained from steering_direction_test.py [0, 1]

        if angle_target < 0 and current_pressure >= 100:
            bend_dir = -bend_dir
        else:
            bend_dir = bend_dir
        response = await self.send_request(NPoints, bend_dir, path)
        distance_to_cont = np.sqrt((response.distance_x)**2 + (response.distance_y)**2)
        self.get_logger().info('Dist'+str(distance_to_cont) + 'Angle'+str(angle_target_deg) + ' Pressure' + str(press_input))
        feedback_msg.distance = distance_to_cont
        goal_handle.publish_feedback(feedback_msg)

        MAX_PRESS = 170
        # Vision-based Control Loop
        while abs(distance_to_cont) > 30 and press_input < MAX_PRESS:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                pressure_msg = DaqInput()
                pressure_msg.pressure = float(0)
                pressure_msg.contour_distance = [0, 0]
                self.daq_pub.publish(pressure_msg)
                return Steer.Result()
            if press_input > MAX_PRESS:
                press_input = MAX_PRESS
                continue
            pressure_msg = DaqInput()
            pressure_msg.pressure = float(press_input)
            pressure_msg.contour_distance = [response.distance_x, response.distance_y]
            self.daq_pub.publish(pressure_msg)
            time.sleep(0.5)
            p_response = await self.send_pressure_request(True)
            current_pressure = p_response.pressure
            bend_angle = p_response.bend_angle
            tip_pos_bframe = p_response.tip_position_baseframe
            tip_ori_bframe = p_response.tip_orientation_baseframe
            self.get_logger().info('Pressure:'+str(current_pressure) + 'angle' + str(bend_angle) + 'input ' + str(press_input))
            response = await self.send_request(NPoints, bend_dir, path)
            distance_to_cont = np.sqrt((response.distance_x)**2 + (response.distance_y)**2)
            feedback_msg.angle_feedback = bend_angle
            feedback_msg.distance = distance_to_cont
            feedback_msg.pressure = current_pressure
            goal_handle.publish_feedback(feedback_msg)
            # change bend direction based on location of contour
            deltaP = 0
            if (np.sign(bend_dir) != np.sign([response.distance_x, response.distance_y])).all():  #distance_to_cont < 0:
                deltaP = -2
                if press_input < 2:  # stops the action before full depressurization
                    break
            else:
                deltaP = 2
            #deltaP = 2 * bend_dir[1]  # change pressure by 2kPa in desired direction
            press_input += deltaP

        goal_handle.succeed()

        result = Steer.Result()
        result.tip_pose.x = tip_pos_bframe[0]
        result.tip_pose.y = tip_pos_bframe[1]
        result.tip_pose.z = tip_pos_bframe[2]

        return result

def main(args=None):
    rclpy.init(args=args)
    steeringaction = SteeringAction()
    rclpy.spin(steeringaction)
    steeringaction.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
