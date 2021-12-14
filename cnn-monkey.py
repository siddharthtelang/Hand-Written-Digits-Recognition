
from cnn import display_report
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import argparse
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from helper_functions import *


def display_report(model, test_data):
    print('\n')
    pred = model.predict(test_data, verbose=1)
    pred = np.argmax(pred, axis=1)

    conf_mat = confusion_matrix(y_true=test_data.labels, y_pred=pred)
    plt.imshow(conf_mat, cmap=plt.cm.Accent)
    indexes = np.arange(len(categories))
    for i in indexes:
        for j in indexes:
            plt.text(j, i, conf_mat[i, j], 
                    horizontalalignment='center', 
                    verticalalignment='center')

    plt.colorbar()
    plt.xticks(indexes, categories, rotation=90)   
    plt.xlabel('Predicted label')
    plt.yticks(indexes, categories)
    plt.ylabel('Ground truth')
    plt.title('Confusion matrix')
    plt.show()


    ground = test_data.labels
    print('\n')
    print(classification_report(ground,pred))

def get_model():
    model = Sequential([
        layers.Conv2D(32, 3, activation='relu', strides=(2,2), input_shape=(height, width, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, strides=(2,2), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(64, 3, strides=(2,2), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, strides=(2,2), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', '--epoch',required=False, default=50, type=int)
    parser.add_argument('-dir', '--dir',required=False, default='Dataset/Monkey_Dataset', type=str)
    args = vars(parser.parse_args())
    epochs = args['epoch']
    root = args['dir']

    height = 224
    width = 224
    batch_size = 32

    training_data, validation_data, test_data, categories =  get_training_testing_data_monkey(root, height, width, batch_size)
    n_categories = len(categories)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

    model = get_model()

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    model.summary()

    history = model.fit(training_data, validation_data=validation_data, epochs=epochs)

    plot_learning_curves(history)

    score = model.evaluate(test_data)

    display_report(model, test_data)
