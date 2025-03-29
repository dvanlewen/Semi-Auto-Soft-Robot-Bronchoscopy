import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from bronchoscopy_interfaces.msg import DaqOutput
from bronchoscopy_interfaces.msg import Path, Tracker
from std_msgs.msg import Bool
import xlsxwriter
import os
from datetime import datetime
import time


class DataSub(Node):
    def __init__(self):
        super().__init__('datasub')
        #self.subscriber = self.create_subscription(Twist, 'UR5e_motion', self.motion_callback, 10)
        self.aurora_sub = self.create_subscription(PoseArray, 'aurora_sensor', self.write_data,10)
        self.press_sub = self.create_subscription(DaqOutput, 'pressure', self.update_pressure, 10)
        self.path_sub = self.create_subscription(Path, 'projection_points', self.update_indices, 10)
        self.data_sub = self.create_subscription(Bool, 'log_data',self.collect_data, 5)
        self.error_sub = self.create_subscription(Tracker, 'tracker_error', self.update_error, 10)
        self.pressure = 0
        self.base_idx = 0
        self.tip_idx = 0
        self.dist_to_cont = [999, 999]

        protonum = input('Enter prototype number:')
        trialnum = input('Enter trial number:')
        path = 'C:/Users/Daniel/Desktop/Results/'
        curr_datetime = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
        datapath = path + curr_datetime
        os.makedirs(datapath)
        self.workbook = xlsxwriter.Workbook(datapath + '/' + datetime.now().strftime('%Y%m%d') + '_P' + protonum + 'T' + trialnum + '.xlsx', {"nan_inf_to_errors": True})
        self.NavData = self.workbook.add_worksheet()
        labels = ['Camera Frame','Pressure','TrackerX','TrackerY','TrackerZ','qBP_x','qBP_y', 'qBP_z', 'qBP_w',
                  'TipX','TipY','TipZ', 'qPT_x', 'qPT_y', 'qPT_z', 'qPT_w', 'Contour_Distance_X', 'Contour_Distance_Y', 'Base_ID', 'Tip_ID','Reference Error', 'Probe Error']
        for x in range(len(labels)):
            self.NavData.write(0, x, labels[x])
        self.row = 1
        self.log_data = False
        self.ref_error = 0
        self.probe_error = 0

    def write_data(self, msg):
        if self.log_data:
            base_pos_pframe = [msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z]
            q_base2patient = [msg.poses[0].orientation.x,
                              msg.poses[0].orientation.y,
                              msg.poses[0].orientation.z,
                              msg.poses[0].orientation.w]

            tip_pos_pframe = [msg.poses[1].position.x, msg.poses[1].position.y, msg.poses[1].position.z]
            q_patient2tip = [msg.poses[1].orientation.x,
                             msg.poses[1].orientation.y,
                             msg.poses[1].orientation.z,
                             msg.poses[1].orientation.w]

            col = 0
            self.NavData.write(self.row, col, time.time() - self.start_time)
            col += 1

            self.NavData.write(self.row, col, self.pressure)
            col += 1

            self.NavData.write(self.row, col, base_pos_pframe[0])
            self.NavData.write(self.row, col+1, base_pos_pframe[1])
            self.NavData.write(self.row, col+2, base_pos_pframe[2])
            col += 3
            self.NavData.write(self.row, col, q_base2patient[0])
            self.NavData.write(self.row, col+1, q_base2patient[1])
            self.NavData.write(self.row, col+2, q_base2patient[2])
            self.NavData.write(self.row, col+3, q_base2patient[3])
            col += 4
            self.NavData.write(self.row, col, tip_pos_pframe[0])
            self.NavData.write(self.row, col+1, tip_pos_pframe[1])
            self.NavData.write(self.row, col+2, tip_pos_pframe[2])
            col += 3
            self.NavData.write(self.row, col, q_patient2tip[0])
            self.NavData.write(self.row, col+1, q_patient2tip[1])
            self.NavData.write(self.row, col+2, q_patient2tip[2])
            self.NavData.write(self.row, col+3, q_patient2tip[3])
            col += 4
            self.NavData.write(self.row, col, self.dist_to_cont[0])
            self.NavData.write(self.row, col+1, self.dist_to_cont[1])
            col += 2
            self.NavData.write(self.row, col, self.base_idx)
            self.NavData.write(self.row, col+1, self.tip_idx)
            col += 2
            self.NavData.write(self.row, col, self.ref_error)
            self.NavData.write(self.row, col+1, self.probe_error)
            self.row += 1

    def update_pressure(self, msg):
        self.pressure = msg.pressure
        self.dist_to_cont = [msg.contour_distance[0], msg.contour_distance[1]]

    def collect_data(self, msg):
        self.log_data = msg
        self.start_time = time.time()

    def update_indices(self, msg):
        self.base_idx = msg.base_index
        self.tip_idx = msg.tip_index

    def update_error(self, msg):
        self.ref_error = msg.tracker_errors[0]
        self.probe_error = msg.tracker_errors[1]


def main(args=None):
    try:
        rclpy.init(args=args)
        datasub = DataSub()
        rclpy.spin(datasub)
    except KeyboardInterrupt:
        datasub.workbook.close()
    datasub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
