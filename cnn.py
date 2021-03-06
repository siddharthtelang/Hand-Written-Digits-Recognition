from tensorflow.keras import models
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, experimental, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import *
from sklearn.metrics import classification_report


def get_model():
    """Get the CNN model

    Returns:
        model: Tensorflow model
    """
    model = Sequential([
        layers.Conv2D(24, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(24, 3, activation='relu'),
        layers.Dropout(0.5),
        layers.MaxPooling2D(strides=(2,2)),
        layers.Conv2D(48, 3, activation='relu'),
        layers.Conv2D(48, 3, activation='relu'),
        layers.Conv2D(48, 3, activation='relu'),
        layers.Dropout(0.5),
        layers.MaxPooling2D(strides=(2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


def display_report(model, test, y_test):
    """Display report

    Args:
        model (tensorflow model): model
        test (ndarray): test data set
        y_test (ndarray): test data labels
    """
    print('\n')
    pred = model.predict(test)
    pred = np.argmax(pred, axis=1)

    category = [i for i in range(10)]
    conf_mat = confusion_matrix(y_test, pred)
    indexes = np.arange(len(category))
    for i in indexes:
        for j in indexes:
            plt.text(j, i, conf_mat[i, j], 
                    horizontalalignment='center', 
                    verticalalignment='center')
    plt.imshow(conf_mat, cmap=plt.cm.Accent)
    plt.colorbar()
    plt.xticks(indexes, category, rotation=90)   
    plt.xlabel('Predicted label')
    plt.yticks(indexes, category)
    plt.ylabel('Ground truth')
    plt.title('Confusion matrix')
    plt.show()

    ground = y_test
    print('\n')
    print(classification_report(ground,pred))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', '--epoch',required=False, default=30, type=int)
    args = vars(parser.parse_args())
    epochs = args['epoch']

    training_data, y_train, testing_data, y_test = get_training_testing_data()

    height = 28
    width = 28

    train = training_data.reshape((training_data.shape[0], height, width, 1))
    test = testing_data.reshape((testing_data.shape[0], height, width, 1))

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model = get_model()

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model.summary()

    history = model.fit(x=train, y=y_train, validation_split=0.3, epochs=epochs, callbacks=[callback])

    plot_learning_curves(history)

    print('\nTest accuracy: \n')
    score = model.evaluate(test, y_test)


    display_report(model, test, y_test)
