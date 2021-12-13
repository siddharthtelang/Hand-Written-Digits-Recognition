import idx2numpy
from dimension_reduction import *


def get_training_testing_data():
    """Get training and testing data

    Returns:
        ndarray: training and testing data
    """
    test_file = 'Dataset/t10k-images.idx3-ubyte'
    test_label_file = 'Dataset/t10k-labels.idx1-ubyte'
    train_file = 'Dataset/train-images.idx3-ubyte'
    train_label_file = 'Dataset/train-labels.idx1-ubyte'

    training_data = idx2numpy.convert_from_file(train_file)
    y_train = idx2numpy.convert_from_file(train_label_file)
    testing_data = idx2numpy.convert_from_file(test_file)
    y_test = idx2numpy.convert_from_file(test_label_file)

    return training_data, y_train, testing_data, y_test


def reduce_dimension(usePCA, X_train, y_train, X_test, y_test, classes):
    """Reduce dimensions using PCA or MDA

    Args:
        usePCA (bool): use of PCA
        X_train (ndarray): Training data
        y_train (ndarray): Training data labels
        X_test (ndarray): Test data
        y_test (ndarray): Test data label
        classes (int): Total classes

    Returns:
        ndarray: Reduced dimensions of input data
    """
    if usePCA:
        projected_train = doPCA(X_train)
        projected_test = doPCA(X_test, dim=projected_train.shape[1])
    else:
        projected_train = doMDA(X_train, y_train, classes-1)
        projected_test = doMDA(X_test, y_test, classes-1)
    
    return projected_train, projected_test