import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from dimension_reduction import *
from helper_functions import *
import argparse

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-training', '--training',required=False, default=5000, type=int)
    parser.add_argument('-testing', '--testing',required=False, default=1000, type=int)
    parser.add_argument('-kernel', '--kernel',required=False, default='rbf', type=str)
    parser.add_argument('-pca', '--pca',required=False, default=False, type=bool)
    parser.add_argument('-mda', '--mda',required=False, default=True, type=bool)
    parser.add_argument('-gamma', '--gamma',required=False, default='0.05', type=float)
    parser.add_argument('-margin', '--margin',required=False, default='0.5', type=float)
    parser.add_argument('-degree', '--degree',required=False, default='3', type=int)
    args = vars(parser.parse_args())

    usePCA = args['pca']
    useMDA = args['mda']
    train = args['training']
    test = args['testing']
    kernel = args['kernel']
    gamma = args['gamma']
    degree = args['degree']
    margin = args['margin']
    classes = 10

    training_data, y_train, testing_data, y_test = get_training_testing_data()

    # flatten the data
    X_train = np.array([training_data[i].flatten() for i in range(training_data.shape[0])])
    X_test = np.array([testing_data[i].flatten() for i in range(testing_data.shape[0])])

    # dimension reduction
    print('Reducing dimensions....')
    projected_train, projected_test = reduce_dimension(usePCA, X_train, y_train, X_test, y_test, classes)
    
    print('Dimensions reduced, proceed for classification...')

    print('Kernel\tMargin\tGamma\t Degree\tAccuracy')

    svm_classifier(projected_train, projected_test, y_train, y_test,\
                       train, test, kernel, gamma=gamma, margin=margin, degree=degree)


    

