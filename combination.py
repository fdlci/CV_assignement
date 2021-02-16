import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from skimage.io import imread
import pickle
import cv2
import matplotlib.pyplot as plt 
import os 

def saving(filename, file_values):
    """Saves the file"""
    return pickle.dump(file_values, open(filename, 'wb'))

def load(filename):
    """Loads the saved file"""
    return pickle.load(open(filename, "rb"))

def sliding_window(im, window_size, step_size):
    """Slides through an input image"""
    for y in range(0, im.shape[0], step_size[1]):
        for x in range(0, im.shape[1], step_size[0]):
            yield (x, y, im[y: y + window_size[1], x: x + window_size[0]])

def get_desc_bg(bounding_boxes, frame_id):
    """Returns the bounding boxes under the form (f_id, x, y, h, w)
    (it is used for the boxes computed with background subtraction)"""
    desc = []
    for bb in bounding_boxes:
        if bb[0] == frame_id:
            desc.append((frame_id, bb[1], bb[2], np.array([0.5]) , bb[3], bb[4]))
    return desc

def get_desc(detections, frame_id):
    """Returns the detections under the form (f_id, x, y, h, w)
    (it is used for the boxes computed with HOG/SVM)"""
    desc = []
    for d in detections:
        if d[0] == frame_id:
            desc.append((frame_id, d[1], d[2], np.array([1]) , d[3], d[4]))
    return desc

def draw_final(filename, detections, frame_id):
    """Draws the bounding boxes"""

    image = imread(filename, as_gray=True)
    if len(detections)>0:
        for [f_id, bb_id, x_tl, y_tl, w, h] in detections:
            if f_id == frame_id:
                cv2.rectangle(image, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)
        plt.axis("off")
        plt.imshow(image)
        plt.title("Final Detection")
        plt.show()

def background_subtraction(filepath):
    """Computes the bounding boxes using background subtraction 
    (uses OpenCV's already implemented functions to do so)"""

    # Compute the background subtraction
    cap = cv2.VideoCapture(filepath)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgbg = cv2.createBackgroundSubtractorMOG2() 

    bounding_boxes = []
    frames = []
    id_frame = 1

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if ret:

            # gray image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

            # perform background subtraction
            fgmask = fgbg.apply(gray)

            # difference between dilation and erosion
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

            # Increase the white regions
            dilation = cv2.dilate(opening, kernel)

            # Finding the contours
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            minarea = 400
            maxarea = 50000

            frames.append(fgmask)

            # Compute bounding boxes via background suppression
            for i in range(len(contours)):
                # only keep parent contours
                if hierarchy[0, i, 3] == -1:
                    area = cv2.contourArea(contours[i])
                    if minarea < area < maxarea:
                        cnt = contours[i]
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        bounding_boxes.append([id_frame, x, y, w, h])

        id_frame += 1
    
    return bounding_boxes, frames

def pedestrian_detector_openCV(dir_path):
    """OpenCV implementation for the computation of the bounding boxes"""

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cpt = 0

    all_rectangles = []

    for frame_id, image_path in enumerate(os.listdir(dir_path)):
        if cpt%10 == 0:
            print('Image: ' + str(cpt+1))
        cpt += 1
        image = cv2.imread(os.path.join(dir_path, image_path))
        
        # keep a minimum image size for accurate predictions
        if image.shape[1] < 400: # if image width < 400
            (height, width) = image.shape[:2]
            ratio = width / float(width) # find the width to height ratio
            image = cv2.resize(image, (400, width*ratio)) # resize the image according to the width to height ratio
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects, weights = hog.detectMultiScale(img_gray, winStride=(2, 2), padding=(10, 10), scale=1.02)

        for detect in rects:
            all_rectangles.append((frame_id+1, detect[0], detect[1], detect[2], detect[3]))

    return all_rectangles, frame_id+1


def final_detections(bounding_boxes, all_detections, last_id):
    """Concatenates both detections (from HOG/SVM and from background subtraction)"""

    all_detect = []
    all_pick = []
    detect_frames = []

    for frame_id in range(1, last_id+1):

        detect_bg = get_desc_bg(bounding_boxes, frame_id)
        detect = get_desc(all_detections, frame_id)

        detections = detect_bg + detect
        print(detections)

        # Computes non-max suppression on the concatenation of both detections
        rectangles = np.array([[f_id, x, y, x + w, y + h] for (f_id, x, y, _, w, h) in detections])
        sc = [score[0] for (f_id, x, y, score, w, h) in detections]
        pick = non_max_suppression(rectangles, probs = sc, overlapThresh = 0.5)

        all_detect.append(detections)
        all_pick.append(pick)

        for bb_id, rect in enumerate(pick):
            f_id, x, y, w, h = rect
            detect_frames.append([frame_id, bb_id, x, y, w-x, h-y])

    return detect_frames, all_detect, all_pick

def pedestrians(path, bb_filepath, w, h, n):
    """Main function to find the bounding boxes"""

    # Computed bounding boxes with subtraction of background
    bounding_boxes = background_subtraction(bb_filepath)[0]

    # Computed bounding_boxes with HOG/SVM
    all_detections, last_id = pedestrian_detector_openCV(path)
    print(all_detections)

    # Computes final bounding boxes
    detect_frames, all_detect, all_pick = final_detections(bounding_boxes, all_detections, last_id)

    return detect_frames

path = 'im'
bb_filepath = 'img1/%3d.jpg'
sol = pedestrians(path, bb_filepath, 1,1,1)
print(sol)