import cv2
import rclpy
from rclpy.node import Node
import process_images_tracking
import numpy as np
#from bronchoscopy_msgs.msg import ContourList
import queue
from bronchoscopy_interfaces.srv import TrackContour
from bronchoscopy_interfaces.msg import Path, DaqInput
from std_msgs.msg import Bool


def distance(x_1, y_1, x_2, y_2):
    """
    Calculates the distance between two points
    """

    dist_between_2_points = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
    signed_dist_x = x_2 - x_1
    signed_dist_y = y_2 - y_1
    return signed_dist_x, signed_dist_y


CAMCENTER_X = 200
CAMCENTER_Y = 200
CAMCENTER = [CAMCENTER_X, CAMCENTER_Y]
# setup array to be size 10, initially all points at camera center
points_2d = np.array([[CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y],
                      [CAMCENTER_X, CAMCENTER_Y]
                      ])

contour_list = []
intrinsic = np.array([[191.99180662, 0, 204.51318274], [0, 193.25891975, 189.01769526], [0, 0, 1]], np.float32)
dist_coeffs = np.array([-6.49287362e-02, -4.62105017e-02, 1.85109873e-05, 4.45296366e-03, 1.77491306e-02], np.float32)

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


class CameraPub(Node):
    def __init__(self):
        super().__init__('camerapub')
        #self.publisher = self.create_publisher(ContourList, 'contour_centers', 10)
        self.cap = cv2.VideoCapture(1)  # 1 for MFI camera
        timer_period = 0.0166
        self.timer = self.create_timer(timer_period, self.frame_callback)
        self.get_logger().info("Camera collecting...")
        self.cv_image = queue.Queue(maxsize=10)
        self.counter = 0
        self.subscriber = self.create_subscription(Path, 'projection_points', self.update_projection, 10)
        self.srv = self.create_service(TrackContour, 'track_contour', self.contour_tracking_service)
        self.data_pub = self.create_publisher(Bool, 'log_data', 5)
        self.contour_pub = self.create_publisher(DaqInput, 'daq_input', 10)
        self.data_counter = 0
        self.tracking = 0
        self.projection = False
        self.tracked_contour_pos = [0,0]
        self.rotation_cmd = 0
        self.tracking_initialized = False

    def frame_callback(self):
        ret, frame = self.cap.read()
        #self.cv_image.put(frame)
        self.contour_list, log_data = process_images_tracking.model_track(frame, self.counter, points_2d)
        #self.contour_list, log_data, capture_frame = process_images_tracking.process_frames(frame,points_2d,self.counter, self.tracked_contour_pos)
        self.counter += 1

        # if self.tracking == 1:
        #     if not self.tracking_initialized:
        #         process_images_tracking.initialize_track(frame, self.tracked_contour_pos)
        #         self.tracking_initialized = True
        #     self.tracked_pos, _ = process_images_tracking.track_frames(frame)
        #     #print(self.tracked_pos)
        #     if len(self.tracked_pos) != 0:
        #         self.tracked_pos = np.array([self.tracked_pos[0][0], self.tracked_pos[0][1]])
        if log_data and self.data_counter < 1:
            msg = Bool()
            msg.data = log_data
            self.data_pub.publish(msg)
            self.data_counter += 1

    def update_projection(self, msg):
        #if not self.projection:
            #points_2d = 200 * np.ones((len(msg.path_point_x),2))
            #self.projection = True
        num_points = len(msg.path_point_x)
        #self.get_logger().info('Received: ' + str(msg.path_point_x))
        for p in range(0, num_points):
            points_2d[p,0] = msg.path_point_x[p]  #poses[p].position.x
            points_2d[p,1] = msg.path_point_y[p]  #poses[p].position.y

    def contour_tracking_service(self, request, response):
        nPathPoints = request.n_path_points
        bend_dir = request.bend_direction
        contourpath_dist = np.zeros(len(self.contour_list))
        self.get_logger().info('Tracking contours...')
        cont_idx = 999
        find = False
        rotated_contour = np.array([200, 200]) + np.array([[np.cos(self.rotation_cmd), -np.sin(self.rotation_cmd)], [np.sin(self.rotation_cmd), np.cos(self.rotation_cmd)]]) @ (self.tracked_contour_pos - np.array([200, 200]))
        #print(rotated_contour)
        if self.tracking == 1:  # Continue tracking previous contour
            for c in range(0, len(self.contour_list)):
                if self.contour_list[c].tracking == 1:
                    cont_idx = c
                    find = True
                    print('SET')
                    self.tracked_contour_pos = np.array([self.contour_list[cont_idx].center_x, self.contour_list[cont_idx].center_y])
            if not find:
                self.tracked_contour_pos = rotated_contour
                response.rotation = 0.0
                response.distance_x = 999
                response.distance_y = 999
                cont_idx = 999
                self.tracking = 0
                print('not found')
                return response

        else:  # Find a new contour to track
            path = request.path_projection
            contour_pos_mm = np.zeros((len(self.contour_list),2))
            contour_vecs = np.zeros((len(self.contour_list),2))
            pathcontour_angle = np.zeros((len(self.contour_list),1))
            contourpath_dist_y = np.zeros((len(self.contour_list),1))
            contourpath_dist_x = np.zeros((len(self.contour_list),1))
            camframe_dist = np.zeros((len(self.contour_list),1))
            contour_camframe = np.zeros((len(self.contour_list),3))
            test = np.zeros((len(self.contour_list),1))
            px2mm_conversion = 0.265
            for v in range(0, len(self.contour_list)):
                contour_camframe[v,:] = 0.175 * (np.linalg.inv(intrinsic) @ [self.contour_list[v].center_x, self.contour_list[v].center_y, 1])
                camframe_dist[v] = np.linalg.norm(path - contour_camframe[v,:])  # for 3D points in cam frame
                print('Path in mm: ', path)
                print(' Contour in mm:  ', contour_camframe)
                contour_pos_mm[v] = [px2mm_conversion * self.contour_list[v].center_x, px2mm_conversion * self.contour_list[v].center_y]
                print('Contour pixel to mm: ', contour_pos_mm)
                contourpath_dist_x[v], contourpath_dist_y[v] = distance(points_2d[int(nPathPoints-1)][0], points_2d[int(nPathPoints-1)][1], self.contour_list[v].center_x, self.contour_list[v].center_y)
                #_, contourpath_dist[v] = distance(points_2d[int(nPathPoints/2)][0], points_2d[int(nPathPoints/2)][1], self.contour_list[v].center_x, self.contour_list[v].center_y)
                #centerdist_y[v], centerdist_x[v] = distance(CAMCENTER_X, CAMCENTER_Y, self.contour_list[v].center_x, self.contour_list[v].center_y)
                contour_vecs[v] = np.array([self.contour_list[v].center_x, self.contour_list[v].center_y]) - CAMCENTER
                pathcontour_angle[v] = np.arctan2(path[1] * contour_vecs[v][0] - path[0] * contour_vecs[v][1], path[0] * contour_vecs[v][0] + path[1] * contour_vecs[v][1])
            contourpath_dist = np.sqrt(contourpath_dist_x**2 + contourpath_dist_y**2)
            # Choose contour based on minimum value (distance or angle)
            cont_idx = np.argmin(camframe_dist)  #contourpath_dist, pathcontour_angle, or camframe_dist
            print('Chosen Center: ', self.contour_list[cont_idx].center_x, self.contour_list[cont_idx].center_y)
            print('Not chosen: ', self.contour_list[0].center)
            print("Index :( : ", cont_idx)
            self.contour_list[cont_idx].tracking = 1
            self.tracking = 1
            self.tracked_contour_pos = np.array([self.contour_list[cont_idx].center_x, self.contour_list[cont_idx].center_y])
        # Calculate distance to chosen contour
        contour_ori = self.tracked_contour_pos  #np.array([self.contour_list[cont_idx].center_x, self.contour_list[cont_idx].center_y])
        distance_to_cont_x, distance_to_cont_y = distance(CAMCENTER_X, CAMCENTER_Y, self.tracked_contour_pos[0], self.tracked_contour_pos[1])  #self.contour_list[cont_idx].center_x, self.contour_list[cont_idx].center_y)
        distance_to_cont = np.sqrt((distance_to_cont_x)**2 + (distance_to_cont_y)**2)
        # Calculate rotation angle to chosen contour
        contour_dir = contour_ori - CAMCENTER
        rotation_angle = np.arctan2(bend_dir[1] * contour_dir[0] - bend_dir[0] * contour_dir[1], bend_dir[0] * contour_dir[0] + bend_dir[1] * contour_dir[1])
        # Reset tracking parameters to stop action if contour is lost
        if cont_idx == 999:
            distance_to_cont_x = 0
            distance_to_cont_y = 0
            rotation_angle = 0
        # if abs(rotation_angle) <= 0.5:
        #     self.tracking = 0
        #     self.tracking_initialized = False
        if distance_to_cont <= 30:
            print('yest', distance_to_cont)
            self.tracking = 0
            self.tracking_initialized = False
        # Submit response
        #self.get_logger().info('Contour distances: X: ' + str(distance_to_cont_x) + ' Y: ' + str(distance_to_cont_y))
        self.rotation_cmd = rotation_angle
        response.rotation = rotation_angle
        response.distance_x = float(distance_to_cont_x)
        response.distance_y = float(distance_to_cont_y)
        self.get_logger().info('Contour distances: X: ' + str(response.distance_x) + ' Y: ' + str(response.distance_y) + 'Angle: ' + str(response.rotation))
        contour_data = DaqInput()
        contour_data.contour_distance = [distance_to_cont_x, distance_to_cont_y]
        self.contour_pub.publish(contour_data)
        return response

def main(args=None):
    rclpy.init(args=args)
    camerapub = CameraPub()
    rclpy.spin(camerapub)
    camerapub.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
