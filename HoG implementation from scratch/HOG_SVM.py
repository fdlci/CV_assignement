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

def hog_features(path, sized):

    hog_features = []
    for image_path in os.listdir(path):
        image = imread(os.path.join(path,image_path),as_gray=True)
        if sized:
            image = image[17:145, 16:80]
        # plt.imshow(image)
        # plt.show()
        hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)[0]
        hog_features.append(hog_feature)
    return hog_features

def select_random_hog_negative_samples(path, window_size, num_window):
    hog_features = []
    for image_path in os.listdir(path):
        image = imread(os.path.join(path,image_path), as_gray=True)
        n, p = image.shape
        for i in range(0,num_window):
            x_min = random.randint(0,n-window_size[0])
            y_min = random.randint(0,p-window_size[1])
            im = image[x_min:x_min+window_size[0],y_min:y_min+window_size[1]]
            hog_feature = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)[0]
            hog_features.append(hog_feature)
    return hog_features

def train_svm(features, labels):
    clf = svm.SVC(C = 0.01, kernel = 'linear')
    model = clf.fit(features, labels)
    return model

def saving(filename, file_values):
    return pickle.dump(file_values, open(filename, 'wb'))

def load(filename):
    return pickle.load(open(filename, "rb"))

if __name__ == "__main__":

    pos_path = 'VIC_Assignment2/INRIAPerson/96X160H96/Train/pos'
    neg_path = 'VIC_Assignment2/INRIAPerson/Train/neg'

    pos_features = hog_features(pos_path, True)
    neg_features = select_random_hog_negative_samples(neg_path, [128,64], 10)

    features = pos_features + neg_features
    labels =  [1]*len(pos_features) + [0]*len(neg_features)
    print("There are " + str(len(features)) + " training samples with " + str(len(pos_features)) + " positive samples and " + str(len(neg_features)) + " negative samples")

    saving('features.p', features)
    saving('labels.p', labels)

    features = np.float32(features)
    labels = np.array(labels)

    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(features))
    features = features[shuffle]
    labels = labels[shuffle] 

    svm_model = train_svm(features, labels)
    saving('svm_model_final.p', svm_model)


