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


def pedestrian_detector_openCV(dir_path):
    """OpenCV implementation for the computation of the bounding boxes"""

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cpt = 0

    all_rectangles = []

    for frame_id, image_path in enumerate(os.listdir(dir_path)):
        if cpt%20 == 0:
            print('Image: ' + str(cpt+1))
        cpt += 1
        image = cv2.imread(os.path.join(dir_path, image_path))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects, weights = hog.detectMultiScale(img_gray, winStride=(2, 2), padding=(10, 10), scale=1.02)

        for bb_id, detect in enumerate(rects):
            all_rectangles.append((frame_id+1, bb_id, detect[0], detect[1], detect[2], detect[3]))

    return all_rectangles


if __name__ == '__main__':
    path = 'img1'
    sol = pedestrian_detector_openCV(path)
    saving('sol_without_resize.p', sol)
    print(sol)
