import cv2
import numpy as np

CAMCENTER_X = 200
CAMCENTER_Y = 200
FILTER_SIZE = 17
THRESH_VAL = 35
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

from __main__ import Video

def trackbar_threshold_value(val):
    """
    Trackbar for modifying threshold value while camera images are captured
    """
    global THRESH_VAL
    THRESH_VAL = val


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
    from __main__ import datapath
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

    global log_data
    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
        if (not log_data):
            log_data = True
            print('Starting recording')
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
    for _ in range(0, len(contours)):
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
            center_threshold = 20
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


window_title = 'Contour Map'
cv2.namedWindow(window_title)
cv2.moveWindow(window_title,900,0)
window_map = 'Contours'
cv2.namedWindow(window_map)
cv2.moveWindow(window_map,0,0)
window_original = 'preview'
#cv2.namedWindow(window_original)
#cv2.moveWindow(window_original,0,0)
cv2.createTrackbar('Threshold Control', window_title, THRESH_VAL, threshold_val_max, trackbar_threshold_value)
trackbar_threshold_value(THRESH_VAL)
processing_counter = 0

def process_frames(preview_buffer, points_2d, counter):
    global contours, capture_frame, processing_counter, Video
#if processing_counter+1 in preview_buffer.keys():
    frame = preview_buffer.get()
    gpuframe = cv2.UMat(frame)
    gray_img = cv2.cvtColor(gpuframe, cv2.COLOR_BGR2GRAY)
    gray_wo_filter = gray_img  #.copy()

    histeq_wo_filter = cv2.equalizeHist(gray_wo_filter)

    med_filter1 = cv2.medianBlur(histeq_wo_filter, FILTER_SIZE)

    ret, thresh = cv2.threshold(med_filter1, THRESH_VAL, 255, 1)

    contours, heirarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    drawing1 = cv2.UMat(np.uint8(255 * np.ones(frame.shape)))  # have to use frame shape since it has bgr channels
    center_of_mass = calculate_com()
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
        if circularness > 0.6:
            cv2.drawContours(drawing1, [cnt], -1, (255, 0, 0), -1, 8)
        else:
            continue
        if area > 10:
            create_contour_list(center_of_mass[c], counter)

    cv2.circle(drawing1, (int(CAMCENTER_X), int(CAMCENTER_Y)), 8, (0, 0, 255), -1, 8, 0)
    # update contour list with contour areas > 10 px every 10 frames
    if counter > 10:
        #for j in range(len(contours)):
            #a = cv2.contourArea(contours[j])
            #result = cv2.pointPolygonTest(contours[j],(0,0),False)
            #if a > 10:
            #    create_contour_list(center_of_mass[j], counter)
        for c in contour_list:
            cv2.putText(drawing1, str(c.id), (int(c.center_x),int(c.center_y)), cv2.FONT_ITALIC, 0.5, (0, 255, 0))
            if c.tracking == 1:
                cv2.circle(drawing1, (int(c.center_x),int(c.center_y)), 4, (0, 255, 255), -1, 8, 0)
            else:
                cv2.circle(drawing1, (int(c.center_x),int(c.center_y)), 4, (0, 255, 0), -1, 8, 0)

    # Overlay the contours onto the original frames
    add_image = cv2.addWeighted(gpuframe, 0.75, drawing1, 0.25, 0.0)
    # Overlay the path onto the frames
    #undistort_img = cv2.undistort(frame, intrinsic, dist_coeffs)

    cv2.imshow(window_original, gpuframe)
    #cv2.imshow(window_map, drawing1)
    # Visualize frame with contours and path
    path_img = cv2.line(add_image,points_2d[0].flatten().astype(int), points_2d[1].flatten().astype(int), (255, 0, 255), 3)
    for k in range(0, len(points_2d) - 1):
        pathadd_img = cv2.line(add_image, tuple(points_2d[k].flatten().astype(int)), tuple(points_2d[k + 1].flatten().astype(int)), (255, 0, 255), 2)
        path_img = cv2.addWeighted(path_img, 0.8 + k * 0.0125, pathadd_img, 0.2 - k * 0.0125, 0.0)
    # Visualize frame with contours and path
    cv2.imshow(window_title, path_img)
    cv2.setMouseCallback(window_title, callback_func)
    # Save frames
    if log_data:
        fn = "DanTest"
        log_image(fn, capture_frame, path_img)
        capture_frame += 1
    cv2.waitKey(1)
    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        Video.video = False
        #break
    #del preview_buffer[processing_counter+1]
    #while len(preview_buffer) > 1:
    #    index = min(preview_buffer.keys())
    #    del preview_buffer[index]
    #    processing_counter += 1
    #processing_counter += 1

    return Video.video, contour_list, log_data, capture_frame
