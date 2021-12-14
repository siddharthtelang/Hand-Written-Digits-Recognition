
from cnn import display_report
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
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


def get_model(vgg_16_model, n_categories):

    # add dense layers
    dense_layer_1 = layers.Dense(512, activation='relu')
    dense_layer_2 = layers.Dense(256, activation='relu')
    dense_layer_3 = layers.Dense(64, activation='relu')

    # prediction layer - total classes
    prediction_layer = tf.keras.layers.Dense(n_categories, activation='softmax')

    model = models.Sequential([
        vgg_16_model,
        dense_layer_1,
        dense_layer_2,
        dense_layer_3,
        layers.BatchNormalization(),
        prediction_layer
    ])
    return model


def get_pretrained_model(freeze):

    vgg_16_model = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, 3),
                include_top=False, weights='imagenet', pooling='max')
    if not freeze:
        vgg_16_model.trainable = True
        for layer in vgg_16_model.layers[:-3]:
            layer.trainable = False

    else:
        vgg_16_model.trainable = False
    
    return vgg_16_model



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-freeze', '--freeze',required=False, default=False, type=bool)
    parser.add_argument('-epochs', '--epoch',required=False, default=30, type=int)
    parser.add_argument('-dir', '--dir',required=False, default='Dataset/Monkey_Dataset', type=str)
    args = vars(parser.parse_args())
    freeze = args['freeze']
    epochs = args['epoch']
    root = args['dir']
    print(freeze)

    height = 224
    width = 224
    batch_size = 32

    training_data, validation_data, test_data, categories =  get_training_testing_data_monkey(root, height, width, batch_size)
    n_categories = len(categories)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

    vgg_16_model = get_pretrained_model(freeze)

    model = get_model(vgg_16_model, n_categories)

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    model.summary()

    history = model.fit(training_data, validation_data=validation_data, epochs=epochs)

    plot_learning_curves(history)

    print('\nTest accuracy:')
    score = model.evaluate(test_data)

    display_report(model, test_data)
