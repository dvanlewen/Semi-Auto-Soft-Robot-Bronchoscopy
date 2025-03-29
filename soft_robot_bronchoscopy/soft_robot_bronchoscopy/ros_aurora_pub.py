import AuroraRegistration
import rclpy
import yaml
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from bronchoscopy_interfaces.srv import SensePressure
from bronchoscopy_interfaces.msg import DaqOutput, Tracker
import pandas as pd
from scipy import spatial


Bx = pd.read_csv("C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\Bx_BotRight.csv").values
By = pd.read_csv("C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\By_BotRight.csv").values
Bz = pd.read_csv("C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\Bz_BotRight.csv").values
centers = pd.read_csv("C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\center_BotRight.csv").values

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

# modified from tf.transformation.py
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

def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    """
    return np.array((-quaternion[0], -quaternion[1],
                     -quaternion[2], quaternion[3]), dtype=np.float64)

def rotation(v_1, v_2):
    """
    Compute a matrix R that rotates v1 to align with v2.
    v1 and v2 must be length-3 1d numpy arrays.
    """

    # unit vectors
    u = v_1 / np.linalg.norm(v_1)
    r_u = v_2 / np.linalg.norm(v_2)
    # dimension of the space and identity
    dim = u.size
    i = np.identity(dim)
    # the cos angle between the vectors
    c = np.dot(u, r_u)
    # a small number
    eps = 1.0e-10
    if np.abs(c - 1.0) < eps:
        # same direction
        return i
    elif np.abs(c + 1.0) < eps:
        # opposite direction
        return -i
    else:
        # the cross product matrix of a vector to rotate around
        k = np.outer(r_u, u) - np.outer(u, r_u)
        # Rodrigues' formula
        return i + k + (k @ k) / (1 + c)

def quaternion_inverse(quaternion):
    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> numpy.allclose(quaternion_multiply(q0, q1), [0, 0, 0, 1])
    True

    """
    return quaternion_conjugate(quaternion) / np.dot(quaternion, quaternion)

def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> numpy.allclose(q, [-44, -14, 48, 28])
    True

    """
    # x0, y0, z0, w0 = quaternion0
    # x1, y1, z1, w1 = quaternion1
    # return np.array((
    #      x1*w0 + y1*z0 - z1*y0 + w1*x0,
    #     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
    #      x1*y0 - y1*x0 + z1*w0 + w1*z0,
    #     -x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=np.float64)
    x0, y0, z0, w0 = quaternion1
    x1, y1, z1, w1 = quaternion0
    return np.array((
        x1*w0 - y1*z0 + z1*y0 + w1*x0,
        x1*z0 + y1*w0 - z1*x0 + w1*y0,
        -x1*y0 + y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=np.float64)

class TrackerPub(Node):
    def __init__(self):
        super().__init__('trackerpub')
        self.publisher = self.create_publisher(PoseArray, 'aurora_sensor', 10)
        timer_period = 0.1  #0.02
        self.timer = self.create_timer(timer_period, self.findPose)
        self.Tracker = AuroraRegistration.EMTrack()

        self.error_pub = self.create_publisher(Tracker, 'tracker_error', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        config = load_config('C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\config.yaml')

        # Rotation from tracker frame to camera frame from initialize_axes.py
        # R_base2tip_static = [[-0.57078967, -0.81704512, 0.08146426],
        #                      [0.8209157, -0.56576767, 0.07748787],
        #                      [-0.01722124, 0.11110457, 0.9936595]]
        R_base2tip_static = [config['tracker2camera'][0], config['tracker2camera'][1],config['tracker2camera'][2]]
        Rot2q = np.append(R_base2tip_static, [[0], [0], [0]], axis=1)
        R_base2tip_static = np.append(Rot2q, [[0, 0, 0, 1]], axis=0)
        self.q_base2tip_static = quaternion_from_matrix(R_base2tip_static)
        self.q_tip2base_static = quaternion_inverse(self.q_base2tip_static)
        # Rotation from bending direction in virtual tip frame to camera frame
        #self.bend_dir = [-0.99796941, 0.06369506]
        self.bend_dir = np.array(config['bendDir_unit'])
        xaxis2bend_dir = np.arctan2(0 * self.bend_dir[0] - 1 * self.bend_dir[1], self.bend_dir[0] * 1 + self.bend_dir[1] * 0)
        #R_bend2tip = rotation([0, 1, 0], [self.bend_dir[0], self.bend_dir[1], 0])  # finds rotation between bend direction and +y direction for CC tip prediction
        #Rotb2q = np.append(R_bend2tip, [[0], [0], [0]], axis=1)
        #R_bend2tip = np.append(Rotb2q, [[0, 0, 0, 1]], axis=0)
        self.q_bend2tip = [np.sin(xaxis2bend_dir/2)*0, np.sin(xaxis2bend_dir/2)*0, np.sin(xaxis2bend_dir/2)*1, np.cos(xaxis2bend_dir/2)]  #quaternion_from_matrix(R_bend2tip)

        # create service client for pressure feedback
        #self.pressure_cli = self.create_client(SensePressure, 'sense_pressure')
        #while not self.pressure_cli.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info('service not available, waiting again...')
        #self.pressure_req = SensePressure.Request()
        self.subscriber = self.create_subscription(DaqOutput, 'pressure', self.collect_pressure, 10)
        self.bend_pressure = 0
        self.tip_pos_bframe = [0, 0, 35]
        self.tip_ori_bframe = [0, 0, 0]
        self.bend_angle = 0

    #def send_pressure_request(self, a):
    #    self.pressure_req.steering = a
    #    self.p_future = self.pressure_cli.call_async(self.pressure_req)
    #    # 4/12 localization gets stuck here
    #    rclpy.spin_until_future_complete(self, self.p_future)
    #    return self.p_future.result()

    def findPose(self):
        msg = PoseArray()
        msg.header.frame_id = 'patient'
        base = Pose()
        tip = Pose()
        base_pos, tracker_error = self.Tracker.getpose()
        base_pos = np.squeeze(base_pos, axis=(2,))
        base_pos = np.squeeze(base_pos, axis=(1,)).tolist()

        error_msg = Tracker()
        error_msg.tracker_errors = tracker_error
        self.error_pub.publish(error_msg)

        base.position.x = base_pos[0]
        base.position.y = base_pos[1]
        base.position.z = base_pos[2]  # Can also edit AuroraRegistration.py to output quaternions

        R_base2patient = self.Tracker.ROTXM2 @ self.Tracker.XYZ_ROT_in_Ref_Disc
        Rot2q2 = np.append(self.Tracker.XYZ_ROT_in_Ref_Disc, [[0], [0], [0]], axis=1)
        R_REF = np.append(Rot2q2, [[0, 0, 0, 1]], axis=0)
        ref_q = quaternion_from_matrix(R_REF)
        Rot2q3 = np.append(self.Tracker.ROTXM2, [[0], [0], [0]], axis=1)
        R_lung = np.append(Rot2q3, [[0, 0, 0, 1]], axis=0)
        lung_q = quaternion_from_matrix(R_lung)
        base_q = quaternion_multiply(lung_q, ref_q)
        print(R_base2patient @ [0,0,5])
        Rot2q = np.append(R_base2patient, [[0], [0], [0]], axis=1)
        R_base2patient = np.append(Rot2q, [[0, 0, 0, 1]], axis=0)
        #base_q = quaternion_from_matrix(R_base2patient)
        #base_q = quaternion_multiply([0,0,1,0], base_q)
        #base_q = quaternion_multiply([1,0,0,0], base_q)
        base.orientation.x = base_q[0]
        base.orientation.y = base_q[1]
        base.orientation.z = base_q[2]
        base.orientation.w = base_q[3]
        self.q_base2patient = [base_q[0],base_q[1],base_q[2],base_q[3]]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'patient'
        t.child_frame_id = 'base'
        t.transform.translation.x = base_pos[0]
        t.transform.translation.y = base_pos[1]
        t.transform.translation.z = base_pos[2]
        t.transform.rotation.x = base_q[0]
        t.transform.rotation.y = base_q[1]
        t.transform.rotation.z = base_q[2]
        t.transform.rotation.w = base_q[3]
        self.tf_broadcaster.sendTransform(t)
        # Localize tip based on constant curvature calculations
        bend_axis = np.cross([self.bend_dir[0],self.bend_dir[1],0],[0,0,-1])
        # if self.bend_pressure > 40:
        #     self.bend_angle = self.bend_angle * 0.9
        q_bend = [np.sin(self.bend_angle/2)*bend_axis[0], np.sin(self.bend_angle/2)*bend_axis[1], np.sin(self.bend_angle/2)*0, np.cos(self.bend_angle/2)]
        
        #Constant Curvature Tip Prediction
        #tip_pos_bendtip = rotate_vector_by_q(quaternion_inverse(q_bend), self.tip_pos_bframe)
        tip_pos_cframe = rotate_vector_by_q(self.q_bend2tip, self.tip_pos_bframe)  #need for base EM probe
        tip_pos_bframe2 = rotate_vector_by_q(self.q_tip2base_static, tip_pos_cframe)
        tip_pos_pframe = rotate_vector_by_q(self.q_base2patient, tip_pos_bframe2)

        # Straight-Angle Tip Prediction
        # tip_pos_cframe = rotate_vector_by_q(self.q_base2tip_static, self.tip_pos_bframe)
        # tip_pos_cframe_bent = rotate_vector_by_q(q_bend, tip_pos_cframe)
        # tip_pos_bframe2 = rotate_vector_by_q(self.q_tip2base_static, tip_pos_cframe_bent)
        # tip_pos_pframe = rotate_vector_by_q(self.q_base2patient, tip_pos_bframe2)

        print('Bframe2', tip_pos_bframe2)
        print('Pframe', tip_pos_pframe)
        #tip_pos_pframe = rotate_vector_by_q([0,0,1,0], tip_pos_pframe)
        tip.position.x = tip_pos_pframe[0] + base_pos[0]
        tip.position.y = tip_pos_pframe[1] + base_pos[1]
        tip.position.z = tip_pos_pframe[2] + base_pos[2]

        #####################################################################
        # Adding the boundaries
        tip_position = [tip.position.x, tip.position.y, tip.position.z]
        i = 0
        path = np.zeros(np.shape(centers))
        for point in centers:
            path[i] = point
            i += 1

        center_tree = spatial.KDTree(path)
        dist2path, ind2path = center_tree.query(tip_position, k=1)
        closedPointCenter = path[ind2path]

        array1 = np.array(Bx[ind2path])
        array2 = np.array(By[ind2path])
        array3 = np.array(Bz[ind2path])
        boundaryPoints = np.stack((array1, array2, array3), axis=1)

        boundaries = np.zeros(np.shape(boundaryPoints))
        i = 0
        for point in boundaryPoints:
            boundaries[i] = point
            i += 1

        boundaries_tree = spatial.KDTree(boundaries)
        dist2bound, ind2bound = boundaries_tree.query(tip_position, k=1)
        closedPointBound = boundaryPoints[ind2bound]

        check_dist = np.sqrt((closedPointCenter[0] - closedPointBound[0])**2 + (closedPointCenter[1] - closedPointBound[1])**2 + (closedPointCenter[2] - closedPointBound[2])**2)

        if dist2path > check_dist:
            tip.position.x = closedPointBound[0]
            tip.position.y = closedPointBound[1]
            tip.position.z = closedPointBound[2]

        #####################################################################

        q_patient2tip = quaternion_multiply(self.q_base2tip_static, quaternion_inverse(self.q_base2patient))  #use below for base EM probe
        q_patient2bend = quaternion_multiply(quaternion_inverse(q_bend), q_patient2tip)  #rotate_vector_by_q(q_base2tip, quaternion_inverse(self.q_base2patient))
        # Note: if bend_angle is 0, the next 3 lines are redundant,
        # q_patient2bend = quaternion_multiply(quaternion_inverse(self.q_bend2tip),q_patient2cam)  #out for tip probe
        # q_patient2bent = quaternion_multiply(q_bend, q_patient2bend)  #out for tip probe
        # q_patient2tip = quaternion_multiply(self.q_bend2tip, q_patient2bent)  #out for tip probe
        #q_patient2tip = quaternion_multiply(q_patient2tip, [0, 0, 1, 0])  #commented out for tip probe  # might need this?, rotates 180 deg about z-axis for path alignment?
        tip.orientation.x = q_patient2bend[0]
        tip.orientation.y = q_patient2bend[1]
        tip.orientation.z = q_patient2bend[2]
        tip.orientation.w = q_patient2bend[3]
        msg.poses = [base, tip]
        self.publisher.publish(msg)
        self.get_logger().info('Tip Pose: ' + str(tip))

        # Add transform broadcaster here to output transform from base to patient frame, derive from Tracker
        t2 = TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = 'tip'
        t2.child_frame_id = 'patient'
        t2.transform.translation.x = tip_pos_pframe[0] + base_pos[0]
        t2.transform.translation.y = tip_pos_pframe[1] + base_pos[1]
        t2.transform.translation.z = tip_pos_pframe[2] + base_pos[2]
        t2.transform.rotation.x = q_patient2tip[0]
        t2.transform.rotation.y = q_patient2tip[1]
        t2.transform.rotation.z = q_patient2tip[2]
        t2.transform.rotation.w = q_patient2tip[3]

        self.tf_broadcaster.sendTransform(t2)

    def collect_pressure(self, msg):
        # Update pressure used in steering DOF
        self.bend_pressure = msg.pressure
        self.tip_pos_bframe = msg.tip_position_baseframe
        self.tip_ori_bframe = msg.tip_orientation_baseframe
        self.bend_angle = msg.bend_angle

def main(args=None):
    rclpy.init(args=args)
    trackerpub = TrackerPub()
    rclpy.spin(trackerpub)
    trackerpub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
