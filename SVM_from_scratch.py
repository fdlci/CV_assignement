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

def pedestrian_detector(path, window_size=[64, 128] , step_size=[10, 10], downscale=1.6, svm_model=load('svm_model_final.p')):
    """Computes the bounding boxes using the HOG/SVM approach"""

    all_detections = []
    print("Computing bounding boxes for image: ")

    for frame_id, image_path in enumerate(os.listdir(path)):
        filename = os.path.join(path, image_path)
        print(filename)
        image = imread(filename, as_gray=True)
        detections = []
        scale = 0

        # Creating a pyramid of different scales, building all possible layers
        for im in pyramid_gaussian(image, downscale = downscale):

            n, p = im.shape
            # if the resized image is smaller than the window-size: stop
            if n < window_size[1] or p < window_size[0]:
                break
            
            for (x,y,im_window) in sliding_window(im, window_size, step_size):
                if im_window.shape[0] != window_size[1] or im_window.shape[1] != window_size[0]:
                    continue
                feature = hog(im_window, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)[0]
                feature = feature.reshape(1,-1)
                prediction = svm_model.predict(feature)

                # If is pedestrian
                if prediction == 1:
                    if svm_model.decision_function(feature) > 0.7:
                        rescale = downscale**scale
                        detections.append((int(x*rescale),int(y*rescale), svm_model.decision_function(feature),
                        int(window_size[0]*rescale), int(window_size[1]*rescale)))

            scale += 1

        # Computing the bounding boxes by getting rid of the overlapping ones
        rectangles = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
        sc = [score[0] for (x, y, score, w, h) in detections]
        pick = non_max_suppression(rectangles, probs = sc, overlapThresh = 0.5)

        for detect in list(pick):
            all_detections.append((frame_id+1, detect[0], detect[1], detect[2]-detect[0], detect[3]-detect[1]))

    return all_detections, frame_id+1

def final_detections(bounding_boxes, all_detections, last_id):
    """Concatenates both detections (from HOG/SVM and from background subtraction)"""

    all_detect = []
    all_pick = []
    detect_frames = []

    for frame_id in range(1, last_id+1):

        detect_bg = get_desc_bg(bounding_boxes, frame_id)
        detect = get_desc(all_detections, frame_id)

        detections = detect_bg + detect

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
    all_detections, last_id = pedestrian_detector(path)

    # Computes final bounding boxes
    detect_frames, all_detect, all_pick = final_detections(bounding_boxes, all_detections, last_id)

    return detect_frames