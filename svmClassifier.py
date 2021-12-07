import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from dimension_reduction import *
from helper_functions import *

def svm_classifier(projected_train, projected_test, y_train, y_test,\
                   train, test, kernel, gamma=0.05, margin=0.5, degree=3):

    acc = []

    Gamma = [0.0000005, 0.0005, 0.05, 0.5]
    C = [0.5, 0.6, 0.8, 1]
    Degree = [2, 3, 4, 5]

    # Gamma = [gamma]
    # C = [margin]
    # Degree = [degree]

    if kernel == 'linear' or kernel == 'sigmoid':
        for c in C:
            acc.clear()
            model = SVC(kernel=kernel, C=c)
            model.fit(projected_train[:train], y_train[:train])
            pred = model.predict(projected_test[:test])
            accuracy = (np.where(pred == y_test[:test])[0].shape[0]) * 100 / test
            print(kernel+'\t'+str(c)+'\t     '+'\t      '+'    '+str(accuracy))
            acc.append(accuracy)

    elif kernel == 'rbf' or kernel == 'poly':
        for c in C:
            for gamma, degree in zip(Gamma, Degree):
                acc.clear()
                if kernel == 'rbf':
                    model = SVC(kernel=kernel, C=c, gamma=gamma)
                else:
                    model = SVC(kernel=kernel, C=c, degree=degree)

                model.fit(projected_train[:train], y_train[:train])
                pred = model.predict(projected_test[:test])
                accuracy = (np.where(pred == y_test[:test])[0].shape[0]) * 100 / test
                if kernel == 'poly':
                    print(kernel+'\t'+str(c)+'\t     '+'\t  '+str(degree)+'\t   '+str(accuracy))
                else:
                    print(kernel+'\t'+str(c)+'\t'+str(gamma)+'\t      '+'    '+str(accuracy))
                acc.append(accuracy)


def reduce_dimension(usePCA, X_train, y_train, X_test, y_test):
    if usePCA:
        projected_train = doPCA(X_train)
        projected_test = doPCA(X_test, dim=projected_train.shape[1])
    else:
        projected_train = doMDA(X_train, y_train, classes-1)
        projected_test = doMDA(X_test, y_test, classes-1)
    
    return projected_train, projected_test


if __name__ == '__main__':

    usePCA = False
    useMDA = True
    classes = 10
    train = 20000
    test = 5000
    # kernel = ['poly', 'rbf', 'linear', 'sigmoid']
    kernel = ['poly']
    gamma = 0.05

    training_data, y_train, testing_data, y_test = get_training_testing_data()

    # flatten the data
    X_train = np.array([training_data[i].flatten() for i in range(training_data.shape[0])])
    X_test = np.array([testing_data[i].flatten() for i in range(testing_data.shape[0])])

    # dimension reduction
    print('Reducing dimensions....')
    projected_train, projected_test = reduce_dimension(usePCA, X_train, y_train, X_test, y_test)
    
    print('Dimensions reduced, proceed for classification...')

    print('Kernel\tMargin\tGamma\t Degree\tAccuracy')

    for k in kernel:
        svm_classifier(projected_train, projected_test, y_train, y_test,\
                       train, test, k, gamma=0.05, margin=0.5, degree=3)


    

