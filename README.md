[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Hand-Written-Digits-Recognition
Hand written digit recognition (MINST dataset) using Logistic Regression, SVM, Deep Neural Nets, and Transfer Learning

## Author

### Siddharth Telang (stelang@umd.edu)

## Subject Code
### CMSC828C/ENEE633 Project 2

## Programming language used: Python3+
### Dependencies (to be installed through pip):
```
1) sklearn: pip install sklearn
2) matplotlib: pip install matplotlib
3) numpy: pip install numpy
4) scipy: pip install scipy
5) tensorflow (as per system and graphics driver)
6) pandas: pip install pandas
7) matplotlib: pip install matplotlib
```

## Contents:
```
1) Code files:
- helper_functions.py: helper functions to get the training and testing data; plot learning curves; dimentionality reduction
- LogisticRegressionCLassifier.py
- cnn.py: For hand-written digit classification
- cnn-monkey.py
- cnn-monkey-transfer-learning.py

2) Jupyter - jupyter notebooks for digit classification, and monkey data set CNNs

3) Report
4) Results - all plots
5) models.txt - models experimented with
6) README files
```

## Dataset
- The data set can be downloaded from the below Google Drive link
```
https://drive.google.com/file/d/1Dx19LPN9bCo9OBpq9wDku3oMzrkFG8mP/view?usp=sharing
```
- Copy the zip file in the current folder and extract it

## Steps to run the code:
### Highlights
- Please ensure this to be the current working directory.
- Various commands with different permutations are mentioned below.
- You may use this on the command prompt and terminal.
- A choice of choosing among pca or mda is provided. Feel free to update if required, only one of them can be set to True at a time.
- training and testing size parameter can be altered to test on various training and testing size - Logistic regression and SVM
- number of iterations, step size, and regularization parameter can also be modified for Logistic regression
- kernel, gamma value, degree, and margin can be modified for SVM through command line
- number of epochs(for CNN) can be modified through the command line argument

### Logistic Regression
#### Configurable: training, testing size, step size, pca, mda, iterations, regularization constant (all are optional, defaults already set)
```
- python3 LogisticRegressionCLassifier.py -training 10000 -testing 1000 -mda True
- python3 LogisticRegressionCLassifier.py -training 10000 -testing 1000 -pca True
- python3 LogisticRegressionCLassifier.py -training 10000 -testing 1000 -mda True -step 0.1 -iter 100 -reg 0.01
```

### Kernel SVM
#### Configurable: training, testing size, kernel, pca, mda (all are optional, defaults already set)
```
python3 svmClassifier.py
python3 svmClassifier.py -training 5000 -testing 1000 -kernel 'rbf' -mda True
python3 svmClassifier.py -kernel 'poly' -mda True
python3 svmClassifier.py -kernel 'linear' -mda True
python3 svmClassifier.py -kernel 'sigmoid' -mda True
python3 svmClassifier.py -pca True

```
### CNN for Handwritten digits classification
#### Configurable: epochs (default: 30)
```
python3 cnn.py -epochs 50
```

### Transfer Learning
####  Configurable: epochs (default: 50), freezing/unfreezing layers of trained model
```
Basic CNN:
python3 cnn-monkey.py -epochs 50
```
Trained network without fine tuning
```
python3 cnn-monkey-transfer_learning.py -epochs 50 -freeze True
```
With Fine Tuning
```
python3 cnn-monkey-transfer_learning.py -epochs 30
```

#### Alternatively, you can also check the Jupyter notebooks for CNN - digits and monkey in the jupyter folder


## Github repository
```
https://github.com/siddharthtelang/Hand-Written-Digits-Recognition
```
