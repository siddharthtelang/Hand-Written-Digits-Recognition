import idx2numpy

def get_training_testing_data():
    test_file = 'Dataset/t10k-images.idx3-ubyte'
    test_label_file = 'Dataset/t10k-labels.idx1-ubyte'
    train_file = 'Dataset/train-images.idx3-ubyte'
    train_label_file = 'Dataset/train-labels.idx1-ubyte'

    training_data = idx2numpy.convert_from_file(train_file)
    y_train = idx2numpy.convert_from_file(train_label_file)
    testing_data = idx2numpy.convert_from_file(test_file)
    y_test = idx2numpy.convert_from_file(test_label_file)

    return training_data, y_train, testing_data, y_test