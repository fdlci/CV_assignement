from skimage.feature import hog
from skimage import data, color, exposure
from skimage.io import imread
import cv2
import os
from os import walk
import pickle
import random
import matplotlib.pyplot as plt
from sklearn import svm
import imutils
import numpy as np
from HOG_SVM import select_random_hog_negative_samples, load
from sklearn.metrics import confusion_matrix, f1_score

def building_test_set(path, window_size = [128,64], num_window = 10):
    pos_features = []
    neg_features = []
    pos_path = os.path.join(path, 'pos')
    for image_path in os.listdir(pos_path):
        image = imread(os.path.join(pos_path,image_path),as_gray=True)
        image = image[3:131, 3:67]
        hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)[0]
        pos_features.append(hog_feature)
    neg_path = os.path.join(path, 'neg')
    neg_features += select_random_hog_negative_samples(neg_path, window_size, num_window)
    test = pos_features + neg_features
    labels = [1]*len(pos_features) + [0]*len(neg_features)
    return test, labels

if __name__ == "__main__":

    path = 'VIC_Assignment2/INRIAPerson/testing'
    test, labels = building_test_set(path)
    print(len(labels))
    
    svm_model = load('VIC_Assignment2\svm_model.p')
    pred_labels = svm_model.predict(test)
    acc = svm_model.score(test, labels)
    conf_matrix = confusion_matrix(labels, pred_labels)
    f1_sc = f1_score(labels, pred_labels)
    print('Accuracy: ' + str(acc))
    print(conf_matrix)
    print('F1_score: ' + str(f1_sc))

    # print(pred_labels)
    # print(labels)