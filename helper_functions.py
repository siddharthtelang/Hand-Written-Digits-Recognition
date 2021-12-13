import idx2numpy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dimension_reduction import *
import pathlib
import pandas as pd


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


def plot_learning_curves(history):
    """Plots learning curves

    Args:
        history (tensorflow history): training history
    """
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    axs[0,0].plot(loss)
    axs[0,0].plot(val_loss)
    axs[0,0].title.set_text('Training Loss vs Validation Loss')
    axs[0,0].legend(['Training', 'Validation'])

    axs[0,1].plot(acc)
    axs[0,1].plot(val_acc)
    axs[0,1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[0,1].legend(['Training', 'Validation'])

    axs[1,0].plot(epochs, acc, 'r', label='Training accuracy')
    axs[1,0].plot(epochs, loss, 'b', label='Training Loss')
    axs[1,0].title.set_text('Training accuracy vs Training loss')
    axs[1,0].legend(loc=0)

    axs[1,1].plot(epochs, val_acc, 'r', label='Validation accuracy')
    axs[1,1].plot(epochs, val_loss, 'b', label='Validation Loss')
    axs[1,1].title.set_text('Validation accuracy vs Validation loss')
    axs[1,1].legend(loc=0)
    plt.show()


def get_training_testing_data(root, height, width, batch_size):

    train_directory = pathlib.Path(f'{root}/training/training/')
    test_directory = pathlib.Path(f'{root}/validation/validation/')

    columns = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
    category_df = pd.read_csv(f"{root}/monkey_labels.txt", names=columns, skiprows=1)
    category_df
    categories = category_df['Common Name']

    trdata = ImageDataGenerator(rescale=1. / 255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )
    training_data = trdata.flow_from_directory(
        train_directory,
        target_size=(height, width),
        batch_size=batch_size,
        shuffle=True,
        subset='training',
        class_mode='categorical'
    )

    valdata = ImageDataGenerator(validation_split=0.2,
        rescale=1./255)
    validation_data = valdata.flow_from_directory(
        train_directory,
        target_size=(height, width),
        subset='validation',
        class_mode='categorical'
    )

    testdata = ImageDataGenerator(rescale=1./255)
    test_data = testdata.flow_from_directory(
        test_directory,
        target_size=(height, width),
        class_mode='categorical'
    )

    return training_data, validation_data, test_data, categories
