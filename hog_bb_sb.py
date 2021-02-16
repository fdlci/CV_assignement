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

def background_subtraction(filepath):
    """Computes the black and white frames using background subtraction 
    (uses OpenCV's already implemented functions to do so)"""

    # Compute the background subtraction
    cap = cv2.VideoCapture(filepath)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgbg = cv2.createBackgroundSubtractorMOG2() 

    frames = []

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
            frames.append(opening)

    return frames

def pedestrian_detector_openCV(frames):
    """Computes the bounding boxes of the black and white frames
    using OpenCV's hog detector"""

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cpt = 0

    all_rectangles = []

    for frame_id, frame in enumerate(frames):

        if cpt%20 == 0:
            print('Image: ' + str(cpt+1))
        cpt += 1
        rects, weights = hog.detectMultiScale(frame, winStride=(2, 2), padding=(10, 10), scale=1.02)

        for bb_id, detect in enumerate(rects):
            all_rectangles.append((frame_id+1, bb_id, detect[0], detect[1], detect[2], detect[3]))

    return all_rectangles

if __name__ == "__main__":

    path = 'img1/%3d.jpg'
    frames = background_subtraction(path)
    bb = pedestrian_detector_openCV(frames)
    saving('bb.p', bb)
