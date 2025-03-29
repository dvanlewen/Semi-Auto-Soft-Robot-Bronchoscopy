import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import torch

CAMCENTER_X = 200
CAMCENTER_Y = 200
FILTER_SIZE = 17
THRESH_VAL = 3
threshold_val_max = 255
CAMCENTER = np.array([CAMCENTER_X, -CAMCENTER_Y])
PARAMETER_VALUES = "_" + str(FILTER_SIZE) + "_" + str(THRESH_VAL) + "_"

capture_frame = 1
counter = 0
cont_id = 0
log_data = False

preview_buffer = {}

intrinsic = np.array([[191.99180662, 0, 204.51318274], [0, 193.25891975, 189.01769526], [0, 0, 1]], np.float32)  #np.array([[162.7643, 0, 203.7765], [0, 165.6396, 199.12472], [0, 0, 1]], np.float32)
dist_coeffs = np.array([-6.49287362e-02, -4.62105017e-02, 1.85109873e-05, 4.45296366e-03, 1.77491306e-02], np.float32)  #np.array([0.00388941, -0.05676799, -0.00424024, -0.00193188, 0.01927032], np.float32)

#from __main__ import Video

def trackbar_threshold_value(val):
    """
    Trackbar for modifying threshold value while camera images are captured
    """
    global THRESH_VAL
    THRESH_VAL = 0.1*val

def trackbar_filter_value(val):
    global FILTER_SIZE
    if val % 2 == 0:
        val += 1
    if val == 0:
        val = 1
    FILTER_SIZE = val

def trackbar_t_max_value(val):
    """
    Trackbar for modifying threshold value while camera images are captured
    """
    global T_max
    T_max = val


class Contour:
    """
    Class that defines a contour
    """

    __slots__ = ('id', 'frame_started', 'num_frames', 'last_frame', 'removal_frame', 'elapsed_frames', 'percentage', 'area', 'tracking', 'center_x', 'center_y', 'center')


contour_list = []  # contour()

#FUNCTIONS

def calculate_com():
    """
    Calculate center of mass for each contour
    """

    # Get the moments
    mu = [None] * len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])
    # Get the mass centers
    mc = [None] * len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    return mc

def log_image(fn, cap_frame, image_to_save):
    global datapath
    """
    Save image to file
    """

    if log_data:
        filename = PARAMETER_VALUES + fn + str(cap_frame) + ".png"
        cv2.imwrite(datapath + '/' + filename, image_to_save)

def callback_func(event, x, y, flags, param):
    """
    Begins or stops logging frames when mouse button+CTRL is pressed on window
    """

    global log_data, datapath
    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
        if (not log_data):
            log_data = True
            print('Starting recording')
            path = 'C:/Users/Daniel/Desktop/Results/'
            curr_datetime = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
            datapath = path + curr_datetime
            os.makedirs(datapath)
        else:
            print('Stopping recording')
            log_data = False

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

def create_contour_list(m_c, counter):
    """
    Create a list of Countours of all contours found in a frame
    """

    global cont_id, contour_list
    corner_threshold = 40.0
    for _ in range(0, len(m_c)):
        # remove contours if in the corners of the frame
        if (m_c[0] < corner_threshold) and (m_c[1] < corner_threshold):
            continue
        elif (m_c[0] < corner_threshold) and (m_c[1] > 400 - corner_threshold):
            continue
        elif (m_c[0] > 400 - corner_threshold) and (m_c[1] > 400 - corner_threshold):
            continue
        elif (m_c[0] > 400 - corner_threshold) and (m_c[1] < corner_threshold):
            continue
        if len(contour_list) == 0:
            contour_list.append(Contour())
            contour_list[0].id = 0
            contour_list[0].center_x = m_c[0]
            contour_list[0].center_y = m_c[1]
            contour_list[0].num_frames = 1
            contour_list[0].frame_started = counter
            contour_list[0].last_frame = counter
            contour_list[0].removal_frame = counter + 1
            contour_list[0].elapsed_frames = 1
            contour_list[0].tracking = 0
            contour_list[0].center = m_c
            cont_id += 1
        else:
            iterations = len(contour_list)
            center_threshold = 40
            for ii in range(0, iterations):
                if ((m_c[0] >= contour_list[ii].center_x - center_threshold)
                    & (m_c[0] <= contour_list[ii].center_x + center_threshold)
                    & (m_c[1] >= contour_list[ii].center_y - center_threshold)
                    & (m_c[1] <= contour_list[ii].center_y + center_threshold)):
                    if contour_list[ii].last_frame != counter:
                        contour_list[ii].center_x = m_c[0]
                        contour_list[ii].center_y = m_c[1]
                        contour_list[ii].center = m_c
                        contour_list[ii].num_frames = contour_list[ii].num_frames + 1
                        contour_list[ii].last_frame = counter
                        contour_list[ii].removal_frame = counter + 15
                    break
                elif(ii == iterations - 1) & (m_c[1] > 0):
                    c = Contour()
                    c.id = cont_id
                    c.center_y = m_c[1]
                    c.center_x = m_c[0]
                    c.frame_started = counter
                    c.num_frames = 1
                    c.last_frame = counter
                    c.removal_frame = counter + 15
                    c.elapsed_frames = 1
                    c.tracking = 0
                    c.center = m_c
                    contour_list.append(c)
                    cont_id += 1
                    break
            # Remove contour if not found
        if len(contour_list) != 0:
            for j in range(len(contour_list) - 1, -1, -1):
                if contour_list[j].removal_frame < counter:
                    contour_list.remove(contour_list[j])


window_original = 'preview'
#cv2.namedWindow(window_original)
#cv2.moveWindow(window_original,0,0)
window_title = 'Contour Map'
cv2.namedWindow(window_title)
cv2.moveWindow(window_title,0,200)
window_map = 'Contours'
#cv2.namedWindow(window_map)
#cv2.moveWindow(window_map,0,0)
cv2.createTrackbar('Threshold Control', window_title, THRESH_VAL, 10, trackbar_threshold_value)
cv2.createTrackbar('Filter', window_title, FILTER_SIZE, threshold_val_max, trackbar_filter_value)
trackbar_filter_value(FILTER_SIZE)
trackbar_threshold_value(THRESH_VAL)
processing_counter = 0

def process_frames(preview_buffer, points_2d, counter, tracked_pos):
    global contours, capture_frame, processing_counter  #, Video
    frame = preview_buffer  #.get()
    gpuframe = cv2.UMat(frame)
    gray_img = cv2.cvtColor(gpuframe, cv2.COLOR_BGR2GRAY)
    gray_wo_filter = gray_img  #.copy()

    # create mask to get rid of corner contours that appear due to uneven illumination/gain
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    radius = 250  #min(center) // 2
    cv2.circle(mask, center, radius, 255, -1)

    #lower_color = np.array([150, 50, 50])
    #upper_color = np.array([180, 255, 255])

    hist_img = cv2.equalizeHist(gray_wo_filter)
    med_filter1 = cv2.medianBlur(hist_img, FILTER_SIZE)
    #blurred = cv2.medianBlur(med_filter1, 55)

    #lab_img = cv2.cvtColor(gpuframe, cv2.COLOR_BGR2Lab)
    #l_chan, a, b = cv2.split(lab_img)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    #l_chan = clahe.apply(l_chan)
    #l_chan = cv2.equalizeHist(l_chan)
    #l_chan = cv2.medianBlur(l_chan, FILTER_SIZE)
    #cv2.imshow('clahed',l_chan)
    #_, l_chan_adapThresh = cv2.threshold(l_chan, THRESH_VAL, 255, 1)  #cv2.adaptiveThreshold(l_chan, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,1, 231, 20)
    #cont2, _ = cv2.findContours(image=l_chan_adapThresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    color_mask = cv2.adaptiveThreshold(med_filter1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 1, 227, 20)  #201, 8 #cv2.inRange(hsv_img, lower_color, upper_color)

    ret, thresh = cv2.threshold(med_filter1, THRESH_VAL, 255, 1)
    #thresh = cv2.adaptiveThreshold(med_filter1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 1, 101, 5)
    thresh = cv2.bitwise_and(thresh, mask)  #color_mask

    color = color_mask  #cv2.bitwise_and(color_mask, mask)
    #cv2.imshow('prefilter',color)
    color = cv2.bitwise_or(color, thresh)
    cnt2, heirarchy2 = cv2.findContours(image=color, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow('color', color)

    contours, heirarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    drawing1 = cv2.UMat(np.uint8(255 * np.ones(frame.shape)))  # have to use frame shape since it has bgr channels
    drawing2 = cv2.UMat(np.uint8(255 * np.ones(frame.shape)))
    #drawing3 = cv2.UMat(np.uint8(255 * np.ones(frame.shape)))
    #cv2.drawContours(drawing3, cont2, -1, (255, 0, 0), -1, 8)
    #cv2.imshow('lab',drawing3)
    # Combine adaptive and global thresholding methods
    contours = np.append(contours, cnt2)
    #heirarchy2 = cv2.UMat.get(heirarchy2)[0]
    center_of_mass = calculate_com()

    # Custom contour filtering
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        cnt = contours[c]
        perim = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.021*perim, True)
        k = cv2.isContourConvex(approx)
        if not k:
            continue
        #if perim == 0:
        #    continue
        circularness = 4 * np.pi * area / (perim**2)
        if area > 10 and circularness > 0.7 and area < 40000:
            cv2.drawContours(drawing1, [cnt], -1, (255, 0, 0), -1, 8)
            create_contour_list(center_of_mass[c], counter)
            #cv2.drawContours(drawing1, [contours[c]], -1, (255, 0, 0), -1, 8)

    #Adaptive threshold Testing, comment out when done
    #for c in range(len(cnt2)):
    #    area = cv2.contourArea(cnt2[c])
    #    cnt = cnt2[c]
    #    perim = cv2.arcLength(cnt, True)
    #    approx = cv2.approxPolyDP(cnt, 0.021*perim, True)
    #    k = cv2.isContourConvex(approx)
    #    if not k:
    #        continue
        #if perim == 0:
        #    continue
    #    circularness = 4 * np.pi * area / (perim**2)
    #    if area > 10 and circularness > 0.7 and area < 40000:
    cv2.drawContours(drawing2, cnt2, -1, (255, 0, 0), -1, 8)
        #else:
            #continue
    #cv2.imshow('adaptive', drawing2)
    #drawing3 = cv2.bitwise_and(drawing1, drawing2)
    #cv2.imshow('filtered colors',drawing3)

    cv2.circle(drawing1, (int(CAMCENTER_X), int(CAMCENTER_Y)), 8, (0, 0, 255), -1, 8, 0)
    cv2.circle(drawing1, (int(tracked_pos[0]), int(tracked_pos[1])), 4, (255, 153, 51), -1, 8, 0)
    # update contour list with contour areas > 10 px every 10 frames
    if counter > 10:
        for c in contour_list:
            cv2.putText(drawing1, str(c.id), (int(c.center_x),int(c.center_y)), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
            if c.tracking == 1:
                cv2.circle(drawing1, (int(c.center_x),int(c.center_y)), 4, (0, 255, 255), -1, 8, 0)
            else:
                cv2.circle(drawing1, (int(c.center_x),int(c.center_y)), 4, (0, 255, 0), -1, 8, 0)

    # Overlay the contours onto the original frames
    add_image = cv2.addWeighted(gpuframe, 0.75, drawing1, 0.25, 0.0)
    #undistort_img = cv2.undistort(frame, intrinsic, dist_coeffs)

    cv2.imshow(window_original, gpuframe)
    #cv2.imshow(window_map, drawing1)
    # Visualize frame with contours and path
    path_img = cv2.line(add_image,points_2d[0].flatten().astype(int), points_2d[1].flatten().astype(int), (255, 0, 255), 3)
    for k in range(0, len(points_2d) - 1):
        pathadd_img = cv2.line(add_image, tuple(points_2d[k].flatten().astype(int)), tuple(points_2d[k + 1].flatten().astype(int)), (255, 0, 255), 2)
        path_img = cv2.addWeighted(path_img, 0.7 + k * 0.0125, pathadd_img, 0.3 - k * 0.0125, 0.0)
    cv2.imshow(window_title, path_img)
    cv2.setMouseCallback(window_title, callback_func)
    # Save frames
    if log_data:
        fn = "DanTest"
        log_image(fn, capture_frame, path_img)
        capture_frame += 1
    cv2.waitKey(1)
    # Waits for a user input to quit the application
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #Video.video = False
        #break
    #del preview_buffer[processing_counter+1]
    #while len(preview_buffer) > 1:
    #    index = min(preview_buffer.keys())
    #    del preview_buffer[index]
    #    processing_counter += 1
    #processing_counter += 1

    return contour_list, log_data, capture_frame


#T_max = 255
#cv2.createTrackbar('T_max', window_title, T_max, threshold_val_max, trackbar_t_max_value)
#trackbar_t_max_value(T_max)

class KalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],
                                                  [0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                                 [0,1,0,1],
                                                 [0,0,1,0],
                                                 [0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],
                                                [0,1,0,0],
                                                [0,0,1,0],
                                                [0,0,0,1]], np.float32) * 0.03

    def predict(self):
        pred = self.kalman.predict()
        return (pred[0], pred[1])

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)


# Parameters for optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize trackers
trackers = []

# Initialize tracking points
#old_points = []
prev_gray = None

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Take first frame
def initialize_track(old_frame, contour_ori):
    global prev_gray, des1, old_points, tracker
    prev_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Initialize SIFT detector and BFMatcher for re-identification
    kp1, des1 = sift.detectAndCompute(prev_gray, None)
    old_points = []
    old_points.append(cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params))
    print('OLD ',old_points)
    if len(np.shape(contour_ori)) == 1:
        contour_ori = [contour_ori]
    #for contours in contour_ori:
    trackers.append(KalmanTracker())
        #old_points.append(np.array([[contours[0], contours[1]]], dtype=np.float32))

def track_frames(frame):
    global contours, prev_gray, old_points, tracker, lk_params, bf, sift, des1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #if len(old_points) == 0:
        # Detect initial points to track
        #contours, _ = cv2.findContours(cv2.adaptiveThreshold(cv2.GaussianBlur(frame_gray, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #for contour in contours:
            #M = cv2.moments(contour)
            #if M['m00'] != 0:
            #cx = int(M['m10'] / M['m00'])
            #cy = int(M['m01'] / M['m00'])
            #if contour.tracking == 1:
            #    old_points.append(np.array([[contour.center_x, contour.center_y]], dtype=np.float32))
        #trackers.append(KalmanTracker())
    #else:
    # Calculate optical flow
    new_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, np.array(old_points[0], dtype=np.float32), None, **lk_params)
    good_new = new_points[st == 1]
    good_old = np.array(old_points[0])[st == 1]
    print('new ',new_points)
    print('old ',old_points)
    prediction = good_new
    # Update trackers and draw
    updated_points = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        #trackers[i].correct(a, b)
        #prediction = trackers[i].predict()
        #cv2.line(frame, (int(prediction[0]), int(prediction[1])), (a, b), (0, 255, 0), 2)
        cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
        updated_points.append((a, b))

    # Re-identification using SIFT and BFMatcher if points are lost
    if len(good_new) < len(old_points):
        kp2, des2 = sift.detectAndCompute(frame_gray, None)
        print(des1)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matched_points = [kp2[m.trainIdx].pt for m in matches[:len(good_old) - len(good_new)]]
        for pt in matched_points:
            updated_points.append(np.array([[pt[0], pt[1]]], dtype=np.float32))

        old_points = updated_points

    prev_gray = frame_gray.copy()

    cv2.imshow('Frame', frame)
    #if cv2.waitKey(30) & 0xFF == ord('q'):
        #break

    return updated_points, prediction


# using trained model to detect the contour

def model_track(frame, counter, points_2d):
    global capture_frame
    model_path = " C:\\Users\\Daniel\\Desktop\\Soft-Robot-Bronchoscopy\\utilities\\best_0806.pt"
    # Load a model
    raw_frame = cv2.UMat(frame)
    device = torch.cuda.set_device(0)
    model = YOLO(model_path)  # load a custom model
    model.to(device)
    cv2.setMouseCallback(window_title, callback_func)

    results = model(frame, verbose=False)[0]

    threshold = THRESH_VAL  #0.3  # we can adjust this value

    contours = []  # save the centroid of detected contours
    frame = cv2.UMat(frame)
    for result in results.boxes.data.tolist():

        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

            # Calculate the center of the bounding box
            width = abs(x1 - x2)
            height = abs(y1 - y2)
            area = width * height
            if area < 400:
                continue

            x = x1
            y = y1
            Center_x = int(x + (width / 2))
            Center_y = int(y + (height / 2))
            center_coordinates = [Center_x, Center_y]   # in pixel
            create_contour_list(center_coordinates, counter)
            frame = cv2.circle(frame, (Center_x, Center_y), 10, (0, 0, 255), -1)
            #cv2.circle(frame, (int(tracked_pos[0]), int(tracked_pos[1])), 4, (255, 153, 51), -1, 8, 0)
            contours.append(center_coordinates)
    for c in contour_list:
        cv2.putText(frame, str(c.id), (int(c.center_x),int(c.center_y)), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
        if c.tracking == 1:
            cv2.circle(frame, (int(c.center_x),int(c.center_y)), 4, (0, 255, 255), -1, 8, 0)
        #else:
            #cv2.circle(frame, (int(c.center_x),int(c.center_y)), 4, (0, 255, 0), -1, 8, 0)

    path_img = cv2.line(frame,points_2d[0].flatten().astype(int), points_2d[1].flatten().astype(int), (255, 0, 255), 3)
    for k in range(0, len(points_2d) - 1):
        pathadd_img = cv2.line(frame, tuple(points_2d[k].flatten().astype(int)), tuple(points_2d[k + 1].flatten().astype(int)), (255, 0, 255), 2)
        path_img = cv2.addWeighted(path_img, 0.7 + k * 0.0125, pathadd_img, 0.3 - k * 0.0125, 0.0)
    if log_data:
        fn = "DanTest"
        log_image(fn, capture_frame, path_img)
        #fn2 = "Raw"
        #log_image(fn2, capture_frame, raw_frame)
        capture_frame += 1

    #rot_frame = cv2.rotate(path_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow(window_title, path_img)
    cv2.waitKey(1)
    return contour_list, log_data
