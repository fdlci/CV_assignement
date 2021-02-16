from skimage.io import imread
import pickle
import cv2
import matplotlib.pyplot as plt 

def saving(filename, file_values):
    """Saves the file"""
    return pickle.dump(file_values, open(filename, 'wb'))

def load(filename):
    """Loads the saved file"""
    return pickle.load(open(filename, "rb"))

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

def pedestrians(filepath, w, h, n):
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
            bb_id = 1
            for i in range(len(contours)):
                # only keep parent contours
                if hierarchy[0, i, 3] == -1:
                    area = cv2.contourArea(contours[i])
                    if minarea < area < maxarea:
                        cnt = contours[i]
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        bounding_boxes.append([id_frame, bb_id, x, y, w, h])
                        bb_id += 1

        id_frame += 1
    
    return bounding_boxes