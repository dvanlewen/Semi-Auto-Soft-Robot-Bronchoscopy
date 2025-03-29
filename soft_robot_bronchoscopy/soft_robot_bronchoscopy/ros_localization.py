import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import scipy.io
from scipy import spatial
import numpy as np
import cv2
import yaml
from rclpy.action import ActionClient
from bronchoscopy_interfaces.action import Steer
from bronchoscopy_interfaces.action import Arm
from bronchoscopy_interfaces.msg import DaqInput
from bronchoscopy_interfaces.msg import Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import Bool
from action_msgs.msg import GoalStatus

_EPS = np.finfo(float).eps * 4.0

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / np.sqrt(t * M[3, 3])
    return q

def rotate_vector_by_q(q, vec):
    """
    Rotate a 3D vector by the quaternion, q
    """
    if len(vec) == 3:
        p = np.append(np.array(vec), 0)
    else:
        p = vec
    q_pr = [-q[0], -q[1], -q[2], q[3]]
    H1 = [q[3]*p[0] + q[0]*p[3] + q[1]*p[2] - q[2]*p[1],
          q[3]*p[1] - q[0]*p[2] + q[1]*p[3] + q[2]*p[0],
          q[3]*p[2] + q[0]*p[1] - q[1]*p[0] + q[2]*p[3],
          q[3]*p[3] - q[0]*p[0] - q[1]*p[1] - q[2]*p[2]]

    H2 = [H1[3]*q_pr[0] + H1[0]*q_pr[3] + H1[1]*q_pr[2] - H1[2]*q_pr[1],
          H1[3]*q_pr[1] - H1[0]*q_pr[2] + H1[1]*q_pr[3] + H1[2]*q_pr[0],
          H1[3]*q_pr[2] + H1[0]*q_pr[1] - H1[1]*q_pr[0] + H1[2]*q_pr[3],
          H1[3]*q_pr[3] - H1[0]*q_pr[0] - H1[1]*q_pr[1] - H1[2]*q_pr[2]]

    if len(vec) == 3:
        p_pr = [H2[0], H2[1], H2[2]]
    else:
        p_pr = [H2[0], H2[1], H2[2], H2[3]]

    return p_pr

def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    """
    return np.array((-quaternion[0], -quaternion[1],
                     -quaternion[2], quaternion[3]), dtype=np.float64)

def quaternion_inverse(quaternion):
    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> numpy.allclose(quaternion_multiply(q0, q1), [0, 0, 0, 1])
    True

    """
    return quaternion_conjugate(quaternion) / np.dot(quaternion, quaternion)

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> numpy.allclose(q, [-44, -14, 48, 28])
    True

    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=np.float64)

# Camera Parameters
intrinsic = np.array([[191.99180662, 0, 204.51318274], [0, 193.25891975, 189.01769526], [0, 0, 1]], np.float32)  #np.array([[162.7643, 0, 203.7765], [0, 165.6396, 199.12472], [0, 0, 1]], np.float32)
dist_coeffs = np.array([-6.49287362e-02, -4.62105017e-02, 1.85109873e-05, 4.45296366e-03, 1.77491306e-02], np.float32)

class Localization(Node):
    def __init__(self):
        super().__init__('localization')
        self.subscriber = self.create_subscription(PoseArray, 'aurora_sensor', self.base_pose_callback, 10)
        # Load Path Planning Variables
        mat = scipy.io.loadmat('C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\newPaths1.mat', mat_dtype=True)
        loadpath = 'BotRight'
        original_path = mat['xPath_' + loadpath]
        self.path = np.zeros(np.shape(original_path))
        i = 0
        for point in original_path:
            point = rotate_vector_by_q([0,0,1,0], point)  # rotate 180 abt z-axis to align path with stl lung model used in test setup
            self.path[i] = point
            i += 1
        lung = mat['lung']
        self.path_tree = spatial.KDTree(self.path)

        # Declare and acquire `target_frame` parameter (base frame)
        self.target_frame = self.declare_parameter('target_frame', 'patient').get_parameter_value().string_value
        self.tip_target_frame = self.declare_parameter('tip_target_frame', 'tip').get_parameter_value().string_value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # create publisher for points projected in camera
        self.publisher = self.create_publisher(Path, 'projection_points', 10)
        #self.timer = self.create_timer(0.5, self.pub_points)
        self.points_2d = np.array([200, 200])

        # This rotation is currently used only in feedback callback as of 7/8/24
        config = load_config('C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\config.yaml')
        R_base2tip_static = [config['tracker2camera'][0], config['tracker2camera'][1],config['tracker2camera'][2]]
        # R_base2tip_static = [[-0.57078967, -0.81704512, 0.08146426],
        #                      [0.8209157, -0.56576767, 0.07748787],
        #                      [-0.01722124, 0.11110457, 0.9936595]]
        Rot2q = np.append(R_base2tip_static, [[0], [0], [0]], axis=1)
        R_base2tip_static = np.append(Rot2q, [[0, 0, 0, 1]], axis=0)
        self.q_base2tip_static = quaternion_from_matrix(R_base2tip_static)
        self.q_tip2base_static = quaternion_inverse(self.q_base2tip_static)
        # R_base2Arm = np.linalg.inv(np.array([[0.0082784, -0.02034058, 0.99975884],
        #                                      [0.99352786, 0.11343441, -0.00591893],
        #                                      [-0.11328666, 0.99333725, 0.02114799]]))  #REPLACE: [1, 0, 0][0, 0, -1][0,1,0]with matrix derived from initialize_axes.py
        self.R_base2Arm = np.linalg.inv(np.array(config['robot2tracker'],dtype=np.float64))

        #bend_dir = np.array([-0.99796941, 0.06369506])  # bending direction vector in camera frame (pixels) from steering_direction_test.py
        self.bend_dir = np.array(config['bendDir_unit'],dtype=np.float64)
        self.branch_idxs = np.array(config[loadpath],dtype=np.float64)
        #branch_idxs = [14, 15, 30, 31, 32, 33]  #Left4 #[14, 15, 38, 39, 41, 42]  #Left5 #[14, 15, 25, 26]  #UpRight1 # [29,30]  #BotRight  
        # create publisher (maybe change to action client) for ur5e
        #self.ur_publisher = self.create_publisher(Twist, 'UR5e_motion', 10)
        #self.daq_pub = self.create_publisher(DaqInput, 'daq_input', 10)

        # create action client for steering
        self.action_client = ActionClient(self, Steer, 'steer')
        self.action_sent = False
        self.armaction_client = ActionClient(self, Arm, 'arm')
        self.arm_action_sent = False
        self.step_action_sent = False
        self.aligned = False
        self.rotation_feedback = 100
        self.dist_to_cont = 100
        self.teleop_on = True

        self.t_subscriber = self.create_subscription(Bool, 'teleoperation', self.teleop_callback, 10)
        self.proj_ind = 999

    def base_pose_callback(self, msg):
        global steering_goal
        #config = load_config('config.yaml')
        from_frame_rel = self.target_frame
        to_frame_rel = 'base'
        try:
            t_base2patient = self.tf_buffer.lookup_transform(to_frame_rel, from_frame_rel,rclpy.time.Time())
            self.q_base2patient = [t_base2patient.transform.rotation.x,
                                   t_base2patient.transform.rotation.y,
                                   t_base2patient.transform.rotation.z,
                                   t_base2patient.transform.rotation.w]
        except:
            self.q_base2patient = [msg.poses[0].orientation.x,
                                   msg.poses[0].orientation.y,
                                   msg.poses[0].orientation.z,
                                   msg.poses[0].orientation.w]

        from_frame_rel_2 = self.tip_target_frame
        to_frame_rel_2 = 'patient'
        try:
            t_patient2tip = self.tf_buffer.lookup_transform(to_frame_rel_2, from_frame_rel_2, rclpy.time.Time())
            q_patient2tip = [t_patient2tip.transform.rotation.x,
                             t_patient2tip.transform.rotation.y,
                             t_patient2tip.transform.rotation.z,
                             t_patient2tip.transform.rotation.w]
        except:
            q_patient2tip = [msg.poses[1].orientation.x,
                             msg.poses[1].orientation.y,
                             msg.poses[1].orientation.z,
                             msg.poses[1].orientation.w]

        base_pos_pframe = [msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z]
        base_ori_pframe = msg.poses[0].orientation
        self.tip_pos_pframe = [msg.poses[1].position.x, msg.poses[1].position.y, msg.poses[1].position.z]  #modified for tip probe
        #self.get_logger().info('Tip Pose: ' + str(tip_pos_pframe))

        # Begin High Level Control
        # Determine closest points on path to the robot
        self.tip_dir_pframe = np.array(rotate_vector_by_q(quaternion_inverse(q_patient2tip), [0, 0, 1]))
        self.tip_dir_pframe = self.tip_dir_pframe / np.linalg.norm(self.tip_dir_pframe)
        forward_projection = 50*self.tip_dir_pframe
        dist, ind = self.path_tree.query(self.tip_pos_pframe)  # + forward_projection)
        basedist2path, base_ind = self.path_tree.query(base_pos_pframe)
        if base_ind >= len(self.path)-10:
            self.path_dir = self.path[len(self.path)-1, :] - self.path[base_ind, :]
        else:
            self.path_dir = self.path[base_ind+10, :] - self.path[base_ind, :]
        # Calculate angle between forward-looking path direction and the robot tip (Could be a service in separate localization node)
        path_dir_base = rotate_vector_by_q(quaternion_inverse(self.q_base2patient), self.path_dir)
        path_dir_base = path_dir_base / np.linalg.norm(path_dir_base)
        path_dir_tip = rotate_vector_by_q(q_patient2tip, self.path_dir)  # path direction in tip (cam) frame
        path_dir_xytipproj = [path_dir_tip[0], path_dir_tip[1]]  # projection of path vector onto xy-plane of tip
        target_rotation_angle = np.arctan2(self.bend_dir[1] * path_dir_xytipproj[0] - self.bend_dir[0] * path_dir_xytipproj[1], self.bend_dir[0] * path_dir_xytipproj[0] + self.bend_dir[1] * path_dir_xytipproj[1])

        q_yzalign = [0, 0, np.sin(-target_rotation_angle/2), np.cos(-target_rotation_angle/2)]
        path_dir_tipyz = rotate_vector_by_q(q_yzalign, path_dir_tip)  # alignment of vector with bend_dir 
        bendyz = np.arctan2(1*self.bend_dir[0]-0*self.bend_dir[1],0*self.bend_dir[0]+1*self.bend_dir[1])
        q_bendyz = [0,0,np.sin(-bendyz/2),np.cos(-bendyz/2)]  # alignment of bend dir with tip y-dir in yz-plane
        path_dir_tipyz = rotate_vector_by_q(q_bendyz, path_dir_tipyz)
        path_dir_yztipproj = [path_dir_tipyz[1], path_dir_tipyz[2]]  # x component in tip should be zero
        target_bend_angle = np.arctan2(1 * path_dir_yztipproj[0] - 0 * path_dir_yztipproj[1], 0 * path_dir_yztipproj[0] + 1 * path_dir_yztipproj[1])

        print('Rotation Angle: ', target_rotation_angle)
        nPathPoints = 30
        path_pt = 25
        if ind+nPathPoints > len(self.path):
            vis_pts_on_path = self.path[ind:len(self.path)-1, :]
            path_pt = len(vis_pts_on_path)-1
        else:
            vis_pts_on_path = self.path[ind:ind + nPathPoints, :]
            path_pt = 25

        # Use current tip orientation to define rotation and translation
        #self.teleop_on = True
        trans = np.zeros(np.shape(vis_pts_on_path))
        points_tipframe = np.zeros(np.shape(vis_pts_on_path))
        points_2d_1 = np.zeros((nPathPoints,2))
        for i in range(0, len(vis_pts_on_path)):
            trans[i,:] = (vis_pts_on_path[i,:] - self.tip_pos_pframe)  #+ [0,0,200] # the path points are pushed into focus length of the camera 200(px) 
            points_tipframe[i] = rotate_vector_by_q(q_patient2tip, trans[i])  # above push could be done after transform but z-axis is roughly same in both frames
            points_2d_1[i,0] = (points_tipframe[i,0] * intrinsic[0,0])/points_tipframe[i,2] + intrinsic[0,2]
            points_2d_1[i,1] = (points_tipframe[i,1] * intrinsic[1,1])/points_tipframe[i,2] + intrinsic[1,2]
        R_patient2tip = quaternion_matrix(quaternion_inverse(q_patient2tip))[:3, :3]
        rotV, _ = cv2.Rodrigues(R_patient2tip[:3][:3])
        rvec = rotV
        #tvec = vis_pts_on_path[0] - tip_pos_pframe
        rotated_tip = rotate_vector_by_q(q_patient2tip, self.tip_pos_pframe)
        tvec = np.array(self.tip_pos_pframe)  #tip_pos_pframe  remove and replace rotated_tip
        points_3d = points_tipframe  #-vis_pts_on_path
        # Map the 3D path to 2D path projected on camera frame
        points_2d, _ = cv2.projectPoints(points_3d, np.zeros((3,1)), np.zeros((3,1)), intrinsic, dist_coeffs)
        # Publish projection points to be used by camera node
        points = Path()
        p_array = []
        p2_array = []
        #print(points_2d)
        for i in range(len(points_2d_1)):
            p = points_2d_1[i][0]
            #p = points_2d[i][0][0]
            p2 = points_2d_1[i][1]
            #p2 = points_2d[i][0][1]
            p_array.append(p)
            p2_array.append(p2)
        points.path_point_x = p_array  # points_tipframe[:][0]
        points.path_point_y = p2_array
        points.tip_index = ind
        points.base_index = base_ind
        self.publisher.publish(points)
        print('Index: ', ind)
        print('Targ Angle:', target_bend_angle)
        if not self.teleop_on:
            # Begin Semi-Auto Low Level Control
            ur5e_goal = Arm.Goal()
            print('Dist: ', dist) #'Tip', self.tip_pos_pframe)
            if abs(target_bend_angle) > 0.05 and not self.aligned and ind in self.branch_idxs and dist < 50:
                # Robot will go through alignment logic before continuing forward
                # Send request to UR5e controller action to rotate robot
                ur5e_goal.path_angle = target_bend_angle
                ur5e_goal.rotation_angle = target_rotation_angle
                ur5e_goal.path_projection = points_tipframe[path_pt,:]  #path_dir_xytipproj
                ur5e_goal.bend_direction = self.bend_dir
                if not self.arm_action_sent:
                    self.get_logger().info('Rotating' + str(target_rotation_angle))
                    self.armaction_client.wait_for_server()
                    self.arm_send_goal_future = self.armaction_client.send_goal_async(ur5e_goal, feedback_callback=self.arm_feedback_callback)
                    self.arm_send_goal_future.add_done_callback(self.arm_goal_response_callback)
                    self.arm_action_sent = True

                # Send request to Steering Action to align tip with contours
                steering_goal = Steer.Goal()
                steering_goal.path_angle = target_bend_angle
                steering_goal.path_projection = points_tipframe[path_pt,:]
                steering_goal.bend_direction = self.bend_dir

                # Send steering action once bending direction is within rotation alignment threshold
                # if not self.action_sent and abs(self.rotation_feedback) < 0.2:
                #     self.get_logger().info('Bending')
                #     self.action_client.wait_for_server()
                #     self._send_goal_future = self.action_client.send_goal_async(steering_goal, feedback_callback=self.feedback_callback)
                #     self._send_goal_future.add_done_callback(self.goal_response_callback)
                #     self.action_sent = True

            else:
                # Robot will align with path in plane of tip and move forward
                if self.action_sent and self.dist_to_cont < 30:
                    self.get_logger().info('Canceling steering to start insertion')
                    self.steering_handle.cancel_goal_async()
                    self.ur5e_handle.cancel_goal_async()
                    self.action_sent = False
                    self.arm_action_sent = False
                    self.dist_to_cont = 100
                # call for insertion action to be performed
                nearest_pt = self.path[base_ind, :]
                basedist2path = -(base_pos_pframe - nearest_pt)

                # if not at a branch point then go through translation logic
                R_base2patient = quaternion_matrix(self.q_base2patient)[:3,:3]
                R_patient2Arm = self.R_base2Arm @ np.linalg.inv(R_base2patient)
                Arm_dists = R_patient2Arm @ basedist2path

                ur5e_goal.displacement = [Arm_dists[0], Arm_dists[2]]
                if not self.arm_action_sent:
                    self.get_logger().info('Translating: ' + str(Arm_dists))
                    self.armaction_client.wait_for_server()
                    self.arm_send_goal_future = self.armaction_client.send_goal_async(ur5e_goal, feedback_callback=self.arm_feedback_callback)
                    self.arm_send_goal_future.add_done_callback(self.arm_goal_response_callback)
                    self.arm_action_sent = True
                if base_ind+1 > len(self.path):
                    next_pt = self.path[base_ind-1,:]
                else:
                    next_pt = self.path[base_ind+1, :]
                dist2nextpt = next_pt - base_pos_pframe
                dist2nextpt = np.sqrt((next_pt[0] - base_pos_pframe[0])**2 + (next_pt[1] - base_pos_pframe[1])**2 + (next_pt[2] - base_pos_pframe[2])**2)
                distperstep = 1  # change this based on gear size
                n_steps = int(dist2nextpt/distperstep)
                if not self.step_action_sent and (self.rotation_feedback == 100 or abs(self.rotation_feedback) < 0.2):
                    step_goal = Steer.Goal()
                    step_goal.n_steps = 1  #n_steps
                    self.get_logger().info('Inserting: ' + str(n_steps))
                    self.action_client.wait_for_server()
                    self._send_goal_future = self.action_client.send_goal_async(step_goal, feedback_callback=self.feedback_callback)
                    self._send_goal_future.add_done_callback(self.goal_response_callback)
                    self.step_action_sent = True
                if ind >= self.proj_ind and self.aligned:
                    steering_goal = Steer.Goal()
                    steering_goal.path_angle = 0
                    steering_goal.path_projection = points_tipframe[path_pt,:]
                    steering_goal.bend_direction = self.bend_dir
                    self.get_logger().info('Release bending')
                    self.action_client.wait_for_server()
                    self._send_goal_future = self.action_client.send_goal_async(steering_goal, feedback_callback=self.feedback_callback)
                    self._send_goal_future.add_done_callback(self.goal_response_callback)
                    self.action_sent = True
                    self.proj_ind = 999
                    self.aligned = False
                    ur5e_goal.displacement = [0, 0]
                    self.armaction_client.wait_for_server()
                    self.arm_send_goal_future = self.armaction_client.send_goal_async(ur5e_goal, feedback_callback=self.arm_feedback_callback)
                    self.arm_send_goal_future.add_done_callback(self.arm_goal_response_callback)
                    self.arm_action_sent = True

        else:
            # Cancel all robot motions if teleoperation toggle is pressed
            if self.action_sent:
                self.get_logger().info('Canceling steering for teleoperation')
                self.steering_handle.cancel_goal_async()
                self.action_sent = False
                self.dist_to_cont = 100
            if self.arm_action_sent:
                self.ur5e_handle.cancel_goal_async()
                self.arm_action_sent = False

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self.steering_handle = goal_handle
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def arm_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('UR5e goal rejected')
            return
        self.get_logger().info('UR5e goal accepted')
        self.ur5e_handle = goal_handle
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_arm_result_callback)

    def get_result_callback(self, future):
        result = future.result().result.tip_pose
        self.action_sent = False
        if future.result().result.steps != 0:
            self.get_logger().info(str(future.result().result.steps))
            self.step_action_sent = False
        elif result.z == 0:
            self.get_logger().info('Bending Released')
        else:
            self.aligned = True
            self.arm_action_sent = False
            self.get_logger().info('Aligned! --- Resulting tip pose:' + str(result))
            forward_projection = 50*self.tip_dir_pframe
            proj_dist, next_ind = self.path_tree.query(self.tip_pos_pframe)
            self.proj_ind = next_ind + 7
            self.get_logger().info('Projection Index: ' + str(self.proj_ind))

    def get_arm_result_callback(self, future):
        global steering_goal
        result = future.result().result.total_rotation
        status = future.result().status
        if status == GoalStatus.STATUS_CANCELED and result == 0.0:
            self.get_logger().info('Arm Action Canceled')
            input('Check Contours')
        self.get_logger().info('Total rotation:' + str(result))
        if abs(self.rotation_feedback) < 0.2:
            rotated = input('Rotation Finished? (y/n)')
            if rotated == 'n':
                self.arm_action_sent = False
                step_goal = Steer.Goal()
                step_goal.n_steps = 1  #n_steps
                self.action_client.wait_for_server()
                self._send_goal_future = self.action_client.send_goal_async(step_goal, feedback_callback=self.feedback_callback)
                self._send_goal_future.add_done_callback(self.goal_response_callback)
                self.step_action_sent = True
            else:
                # Send steering action once bending direction is within rotation alignment threshold
                if not self.action_sent:
                    # step_goal = Steer.Goal()
                    # step_goal.n_steps = 5
                    # self.get_logger().info('Inserting')
                    # self.action_client.wait_for_server()
                    # self._send_goal_future = self.action_client.send_goal_async(step_goal, feedback_callback=self.feedback_callback)

                    self.get_logger().info('Bending: ' + str(steering_goal.path_angle))
                    self.action_client.wait_for_server()
                    self._send_goal_future = self.action_client.send_goal_async(steering_goal, feedback_callback=self.feedback_callback)
                    self._send_goal_future.add_done_callback(self.goal_response_callback)
                    self.action_sent = True
                    self.rotation_feedback = 100
        else:
            self.arm_action_sent = False

    def feedback_callback(self, feedback_msg):
        self.get_logger().info('Receiving steering feedback')
        self.get_logger().info('Remaining distance: ' + str(feedback_msg.feedback.distance) + '  Bend Angle Estimation: ' + str(feedback_msg.feedback.angle_feedback))
        self.get_logger().info('Pressure: ' + str(feedback_msg.feedback.pressure))
        b_angle = feedback_msg.feedback.angle_feedback
        self.dist_to_cont = feedback_msg.feedback.distance
        q_bend = [np.sin(b_angle/2)*1, np.sin(b_angle/2)*0, np.sin(b_angle/2)*0, np.cos(b_angle/2)]
        q_base2tip = quaternion_multiply(self.q_base2tip_static, q_bend)

        q_patient2tip = quaternion_multiply(q_base2tip, quaternion_inverse(self.q_base2patient))

    def arm_feedback_callback(self, feedback_msg):
        self.rotation_feedback = feedback_msg.feedback.rotation_angle
        self.get_logger().info('Remaining rotation:' + str(self.rotation_feedback))

    def teleop_callback(self, msg):
        self.teleop_on = msg.data


def main(args=None):
    rclpy.init(args=args)
    localization = Localization()
    rclpy.spin(localization)
    localization.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
