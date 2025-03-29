#import process_images
import cv2
import time
import numpy as np
from datetime import datetime
import os
import threading
import queue
#from multiprocessing import Queue, Pool



CAMCENTER_X = 200
CAMCENTER_Y = 200
counter = 0
class Video:
    def __init__(self):
        self.video = True

Video = Video()

preview_buffer = {}
path = 'C:/Users/Daniel/Desktop/Results/'
curr_datetime = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
datapath = path + curr_datetime
os.makedirs(datapath)

buffer1 = queue.Queue(maxsize=5)
def init_pool(d_b):
    global buffer1
    buffer1 = d_b
#pool = Pool(2, initializer=init_pool, initargs=(buffer1,))


cap = cv2.VideoCapture(0)
points_2d = np.array([[CAMCENTER_X, CAMCENTER_Y], [CAMCENTER_X, CAMCENTER_Y], [CAMCENTER_X, CAMCENTER_Y]])
_, frame = cap.read()
preview_buffer[counter+1] = frame
def readvid():
    global counter, Video
    while Video.video:

        # Capture frame-by-frame
        _, frame = cap.read()
        preview_buffer[counter+1] = frame
        buffer1.put(frame)

        time.sleep(0.0166666)
        counter += 1

#readit = pool.apply_async(readvid)

t1 = threading.Thread(target=readvid)
t1.start()
def procvid():
    import process_images
    global Video
    while Video.video:
        Video.video, contour_list, _, _ = process_images.process_frames(buffer1, points_2d, counter)

t2 = threading.Thread(target=procvid)
t2.start()

t1.join()
t2.join()