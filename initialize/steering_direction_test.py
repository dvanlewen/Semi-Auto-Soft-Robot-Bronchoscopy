#import process_images
import cv2
import time
import numpy as np
from datetime import datetime
import yaml
import os

import threading
import queue
import scipy.io
from scipy import spatial
import matplotlib.pyplot as plt
from closedloop_steering import pressure_pumps
#from multiprocessing import Queue, Pool


lock = threading.Lock()
CAMCENTER_X = 200
CAMCENTER_Y = 200
CAMCENTER = [CAMCENTER_X, CAMCENTER_Y]
intrinsic = np.array([[191.99180662, 0, 204.51318274], [0, 193.25891975, 189.01769526], [0, 0, 1]], np.float32)  #np.array([[162.7643, 0, 203.7765], [0, 165.6396, 199.12472], [0, 0, 1]], np.float32)
dist_coeffs = np.array([-6.49287362e-02, -4.62105017e-02, 1.85109873e-05, 4.45296366e-03, 1.77491306e-02], np.float32)  #np.array([0.00388941, -0.05676799, -0.00424024, -0.00193188, 0.01927032], np.float32)

contour_list = []
counter = 0
class Video:
    def __init__(self):
        self.video = True

Video = Video()
runthreads = threading.Event()
runthreads.set()

preview_buffer = {}
path = 'C:/Users/Daniel/Desktop/Results/'
curr_datetime = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
datapath = path + curr_datetime

buffer1 = queue.Queue(maxsize=5)
rotateQueue = queue.Queue(maxsize=5)
bendQueue = queue.Queue(maxsize=5)
pathQueue = queue.Queue(maxsize=5)
#pool = Pool(2, initializer=init_pool, initargs=(buffer1,))

# converts numpy array to list for yaml
def convertNumpy2List(data):
    return {k: v.tolist() for k, v in data.items()}

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


def distance(x_1, y_1, x_2, y_2):
    """
    Calculates the distance between two points
    """

    dist_between_2_points = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
    return dist_between_2_points

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


cap = cv2.VideoCapture(1)
points_2d = np.array([[CAMCENTER_X, CAMCENTER_Y], [CAMCENTER_X, CAMCENTER_Y], [CAMCENTER_X, CAMCENTER_Y]])
_, frame = cap.read()
preview_buffer[counter+1] = frame
pathQueue.put(points_2d)
R_rotateDOF = np.identity(3)


def readvid():
    global counter
    while runthreads.is_set():

        # Capture frame-by-frame
        _, frame = cap.read()
        preview_buffer[counter+1] = frame
        buffer1.put(frame)

        time.sleep(0.0166666)
        counter += 1

#readit = pool.apply_async(readvid)


t1 = threading.Thread(target=readvid)

def procvid():
    import process_images
    global Video, contour_list
    while Video.video:
        if pathQueue.empty():
            points = np.array([[CAMCENTER_X, CAMCENTER_Y], [CAMCENTER_X, CAMCENTER_Y], [CAMCENTER_X, CAMCENTER_Y]])
        else:
            points = pathQueue.get()
        Video.video, contour_list, _, _ = process_images.process_frames(buffer1, points, counter)
    runthreads.clear()


t2 = threading.Thread(target=procvid)

t1.start()
t2.start()
input('Press Enter to Begin')

with lock:
    startingLoc = np.array(contour_list[1].center)
    startingID = contour_list[1].id
pressure_pumps.set_pressure(100)
time.sleep(3)
with lock:
    for contour in contour_list:
        if contour.id == startingID:
            bentLoc = np.array(contour.center)

bendDir = startingLoc - bentLoc
bendDir_unit = bendDir/np.linalg.norm(bendDir)
pressure_pumps.pressure_pumps_close()
print('Bend Direction: ',bendDir)
print('Unit Vector: ', bendDir_unit)

# saving to config file
save2file = input('Update calibration data to existing config file? (y/n)')
if save2file == 'y':
    calib_matrices = {
        'bendDir': bendDir,
        'bendDir_unit': bendDir_unit,
    }
    config_path = os.path.join(os.path.dirname(__file__),'..','utilities','config.yaml')
    updateConfig(config_path,calib_matrices)
    print('Calibration data saved to config.yaml')
else:
    print('Data not updated.')


input('Enter to End')

Video.video = False
t1.join()
t2.join()
#cv2.destroyAllWindows()
