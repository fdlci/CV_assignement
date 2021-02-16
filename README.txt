Assignment 2 VIC

Files that can be found in the Hog Implementation from scratch folder:

- HOG_SVM.py: python file used to compute the SVM model. Builds the training set from the INRIA's person
dataset and trains the SVM model using the HOG features.

- test_svm.py: tests the SVM model on a test set to see its performances

Other python files:

- Florez_de_la_colina.py: Computes the bounding boxes using the method described in the report
- openCV_imp.py: computes the bounding boxes only using openCV's hog implementation
- combination.py: computes the bounding boxes by combining background subtraction with HoG/SVM
- hog_bb_sb.py: computes the bounding boxes using HoG over the black and white frames
- SVM_from_scratch.py: My implementation of the bounding boxes using the svm model I trained

Further details can be found in the report.

Important:
- to load the svm_final_model.p: need for scikit-learn==0.23.2
- in assignment2.ipynb, there is a file_path named bb_filepath. It is set to the path for the given dataset.
To test the new dataset, you will need to change this path. In this %3d, the 3 indicates the number of digits
there are in the paths of each image. Thus, this digit will correspond to the new number of digits in the
new dataset.
