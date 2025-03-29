from rtde_rotation import RobotArm
import numpy as np
from AuroraRegistration import EMTrack
import cv2
import threading
import time
import queue
from scipy.spatial.transform import Rotation as R
import yaml
import os
from pathlib import Path

# Initial Localization of Robot Axes

class Video:
    def __init__(self):
        self.video = True


CAMCENTER_X = 200
CAMCENTER_Y = 200
# STATE VARIABLES
Video = Video()
runthreads = threading.Event()
runthreads.set()
moveforward = threading.Event()
lock = threading.Lock()
counter = 0

Queuesize = 10
read_buffer = queue.Queue(maxsize=Queuesize)
rotateQueue = queue.Queue(maxsize=Queuesize)
bendQueue = queue.Queue(maxsize=Queuesize)
pathQueue = queue.Queue(maxsize=Queuesize)
tipQueue = queue.Queue(maxsize=Queuesize)

contour_list = []
intrinsic = np.array([[191.99180662, 0, 204.51318274], [0, 193.25891975, 189.01769526], [0, 0, 1]], np.float32)  #np.array([[162.7643, 0, 203.7765], [0, 165.6396, 199.12472], [0, 0, 1]], np.float32)

# converts numpy array to list for yaml
def convertNumpy2List(data):
    return {k: v.tolist() for k, v in data.items()}

# saves list data to yaml file
def saveConfig(file_path, data):
    serialData = convertNumpy2List(data)
    with open(file_path, 'w') as yaml_file:
        yaml.safe_dump(serialData, yaml_file, default_flow_style=False)

# updates existing yaml file with data to yaml file
def updateConfig(file_path, data):
    serialData = convertNumpy2List(data)
    with open(file_path,'r') as yaml_file:
        existingConfig = yaml.safe_load(yaml_file)
    if hasattr(existingConfig,str(data.keys())):
        existingConfig[data.keys()].update(serialData)
    else:
        existingConfig.update(serialData)
    if existingConfig:
        with open(file_path, 'w') as yaml_file:
            yaml.safe_dump(existingConfig, yaml_file, default_flow_style=False)

# Compute a rotation matrix that aligns vector v1 to vector v2 using scipy's align_vectors method.
def rotation_from_vec(v1, v2):
    # Normalize vectors
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)

    rotation, _ = R.align_vectors([u2,[0,1,0]], [u1,[0,0,1]])
    return rotation.as_matrix()

# Calculate the rotation matrix from robot to tracker using rotation computation.
def calculate_robot_to_tracker_rotation(translation_x, base_diff_baseframe):
    R_rob2base = rotation_from_vec([translation_x[0], 0.0, translation_x[1]], base_diff_baseframe)
    return R_rob2base

# Calculate the rotation matrix from robot to camera using rotation computation.
def calculate_robot_to_camera_rotation(translation_x, contour_trans):
    R_rob2tip = rotation_from_vec([translation_x[0], 0.0, translation_x[1]], [contour_trans[0], contour_trans[1], 0.0])
    return R_rob2tip

def vision_thread():
    global counter  #, preview_buffer, Video

    cap = cv2.VideoCapture(1)

    while runthreads.is_set():
        # Capture frame-by-frame
        _, frame = cap.read()
        #preview_buffer[counter+1] = frame
        read_buffer.put(frame)

        time.sleep(0.0166666)
        counter += 1

def processing_thread():
    import process_images
    global counter, points, preview_buffer, Video, contours, capture_frame, dist_coeffs, intrinsic, contour_list, log_data

    while Video.video:
        if pathQueue.empty():
            points = np.array([[CAMCENTER_X, CAMCENTER_Y], [CAMCENTER_X, CAMCENTER_Y], [CAMCENTER_X, CAMCENTER_Y]])
        else:
            points = pathQueue.get()
        Video.video, contour_list, log_data, capture_frame = process_images.process_frames(read_buffer, points, counter)
    runthreads.clear()


t1 = threading.Thread(target=vision_thread)
t2 = threading.Thread(target=processing_thread)
t1.start()
t2.start()

# Initialize Tracker and Robot Arm
Tracker = EMTrack()
# Get initial location of base sensor and contours in camera
base_pos_patientframe, _ = Tracker.getpose()
base_pos_patientframe = np.squeeze(base_pos_patientframe, axis=(2,))
base_pos_patientframe = np.squeeze(base_pos_patientframe, axis=(1,)) #.tolist()
R_base2patient = Tracker.ROTXM2 @ Tracker.XYZ_ROT_in_Ref_Disc
Robotarm = RobotArm()
input('Press Enter to Begin')
c1 = np.zeros([len(contour_list),2])
c1_ID = np.zeros([len(contour_list),1])
c1_cameraframe = np.zeros([len(contour_list),3])
with lock:
    for i in range(0,len(contour_list)):
        c1[i] = contour_list[i].center
        c1_ID[i] = contour_list[i].id
        c1_cameraframe[i] = 30 * (np.linalg.inv(intrinsic) @ [contour_list[i].center_x, contour_list[i].center_y, 1])


# Move UR5e tool in x-direction
translation_x = [-0.005, -0.005]  # translation is in m
pose = Robotarm.translate_axis(translation_x, [0, 2])
time.sleep(7)
# Get translated values
base_pos_patientframe2, _ = Tracker.getpose()
base_pos_patientframe2 = np.squeeze(base_pos_patientframe2, axis=(2,))
base_pos_patientframe2 = np.squeeze(base_pos_patientframe2, axis=(1,))  #.tolist()
c2 = np.zeros([len(contour_list),2])
c2_ID = np.zeros([len(contour_list), 1])
c2_cameraframe = np.zeros([len(contour_list),3])
with lock:
    for j in range(0, len(contour_list)):
        c2[j] = contour_list[j].center
        c2_ID[j] = contour_list[j].id
        c2_cameraframe[j] = 30 * (np.linalg.inv(intrinsic) @ [contour_list[j].center_x, contour_list[j].center_y, 1])

base_diff_patientframe = base_pos_patientframe2 - base_pos_patientframe
base_diff_baseframe = np.linalg.inv(R_base2patient) @ base_diff_patientframe
print('Translation in Aurora Base frame', base_diff_baseframe)

# Calculate rotations
R_rob2base = calculate_robot_to_tracker_rotation(translation_x*1000, base_diff_baseframe)
print('Rotation from robot to tracker', R_rob2base)

# contour_trans = np.array([0, 0])
# contour_trans_camframe = np.array([0, 0, 0])
# i = 0
# for ID in c1_ID:
#     for cont in range(0, len(c2)):
#         if ID == c2_ID[cont]:
#             contour_trans = (c2[cont] - c1[i])
#             contour_trans_camframe = (c2_cameraframe[cont] - c1_cameraframe[i])
#             i += 1

contour_trans = np.array([0, 0])
i = 0
count = 0
for ID in c1_ID:
    for cont in range(0,len(c2)):
        if ID == c2_ID[cont]:
            contour_trans = -(c2[cont] - c1[count])  # (-) calculates translation of camera relative to contours
            contour_trans_camframe = c2_cameraframe[cont] - c1_cameraframe[count]
            print(c2[cont]-c1[count])
            i += 1
    count += 1

avg_contour_trans = np.sum(contour_trans, 0) / np.shape(contour_trans)[0]
print('Translation of Contours in Camera Frame', contour_trans)
print('Translation of Contours in Cam Frame (mm)', contour_trans_camframe)

R_rob2tip = calculate_robot_to_camera_rotation(translation_x*1000, contour_trans)
print('Rotation from robot to camera', R_rob2tip)

R_base2tip_static = R_rob2tip @ np.linalg.inv(R_rob2base)
print('Rotation from tracker to camera', R_base2tip_static)


# saving to config file
save2file = input('Save calibration data to file? (y/n)')
if save2file == 'y':
    calib_matrices = {
        'robot2tracker': R_rob2base,
        'tracker2camera': R_base2tip_static,
    }
    config_path = os.path.join(os.path.dirname(__file__),'..','utilities','config.yaml')
    #if os.path.exists('config.yaml'):
    updateConfig(config_path,calib_matrices)
    #else:
        #saveConfig(config_path,calib_matrices)
else:
    print('Data not updated.')

Video.video = False
t1.join()
t2.join()
