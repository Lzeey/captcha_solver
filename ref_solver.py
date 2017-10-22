# -*- coding: utf-8 -*-
#@author: Admin
"""
Created on Fri Apr 21 16:29:29 2017

@author: ph
"""
import glob
import logging
import os

import numpy as np
import skimage.io
import sklearn

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint


logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

DEBUG = False
if DEBUG:
    IMAGE_PATH = r'./debugging/*.png'
else:
    IMAGE_PATH = r'./labelled/*.png'

# Constants to map characters to numbers and vice versa
CHARSET = 'ABCDEFGHKLMNPRSTUVWYZabcdefghklmnprstuvwyz23456789'
CHAR_TO_IDX = {value: key for key, value in enumerate(CHARSET)}
IDX_TO_CHAR = {key: value for key, value in enumerate(CHARSET)}

CAPTCHA_LENGTH = 7
NUM_CLASSES = len(CHARSET)
BATCH_SIZE = 64
NUM_EPOCHS = 7

# preprocess images
def preprocess_image(f):
    # convert image to greyscale and cast to float32
    grey_image = skimage.io.imread(f, as_grey=True).astype(np.float32)
    # and inverts the colours
    inverted_image = skimage.util.invert(grey_image)
    # add dimension for colour channel (i.e. greyscale)
    inverted_image = inverted_image.reshape(inverted_image.shape + (1,))

    return inverted_image


def generate_np_labels(labels):
    """
    TODO improve on this documentation
    generate_np_labels builds an numpy array containing indexes of characters in labels
    E.g. ABCDEFG -> [0, 1, 2, 3, 4, 5, 6, 7]
    Args:
        list of string labels
    Returns:
        numpy arrays of labels as unsigned integers (0 to 255)
    """
    np_label = np.zeros((len(labels), CAPTCHA_LENGTH), dtype=np.float32)
    for label_idx, label in enumerate(labels):
        for char_idx, char in enumerate(label):
            np_label[label_idx][char_idx] = CHAR_TO_IDX[char]

    return np_label


def load_dataset(path):
    # load dataset from disk if previously saved
    if next(glob.iglob("*.npy"), None):
        logging.debug("Loading arrays from disk")
        x_train = np.load('x_train.npy')
        y_train = np.load('y_train.npy')
        x_test = np.load('x_test.npy')
        y_test = np.load('y_test.npy')
        logging.debug("Load done")

        return x_train, y_train, x_test, y_test
    else:        
        logging.debug("Loading images from disk")
        # load images into numpy array
        image_collection = skimage.io.collection.ImageCollection(IMAGE_PATH, load_func=preprocess_image)
        images = image_collection.concatenate()
        # load labels
        image_filenames = [os.path.splitext(os.path.basename(p))[0] for p in image_collection.files]
        labels = generate_np_labels(image_filenames)

        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

        np.save('x_train.npy', x_train)
        np.save('y_train.npy', y_train)
        np.save('x_test.npy', x_test)
        np.save('y_test.npy', y_test)
        logging.debug("Load done")

        return x_train, y_train, x_test, y_test

        
x_train, y_train, x_test, y_test = load_dataset(IMAGE_PATH)
x_height = x_train.shape[1]
x_width = x_train.shape[2]
x_channels = x_train.shape[3]


def add_funny_conv_layer(input_t, channel_size, max_pool_size=2):
    x = Conv2D(channel_size, (3, 3), padding='same' )(input_t)
    x = BatchNormalization(axis=3 )(x)
    x = Activation('relu')(x)
    x = Conv2D(channel_size, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(x)

    return x


# create model using Keras
inputs = Input(shape=(x_height, x_width, x_channels))
conv_output = add_funny_conv_layer(inputs, 64)
conv_output = Dropout(0.1)(conv_output)
conv_output = add_funny_conv_layer(conv_output, 128)
conv_output = Dropout(0.1)(conv_output)
conv_output = add_funny_conv_layer(conv_output, 256)
conv_output = Dropout(0.1)(conv_output)
conv_output = add_funny_conv_layer(conv_output, 512)
conv_output = Dropout(0.1)(conv_output)
conv_output = add_funny_conv_layer(conv_output, 512)
conv_output = Dropout(0.1)(conv_output)


flatten = Flatten(name="flatten_conv_output")
conv_output = flatten(conv_output)

loss_list = []
loss_names = ['output' + str(i) for i in range(CAPTCHA_LENGTH)]
for i in range(CAPTCHA_LENGTH):
    dense_output = Dense(NUM_CLASSES, activation='softmax', name=loss_names[i])(conv_output)
    loss_list.append(dense_output)

model = Model(inputs=inputs, outputs=loss_list)
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
#              loss_weights=[1./CAPTCHA_LENGTH]*CAPTCHA_LENGTH,
              loss_weights={
                  'output0': 0.025,
                  'output1': 0.025,
                  'output2': 0.2,
                  'output3': 0.5,
                  'output4': 0.2,
                  'output5': 0.025,
                  'output6': 0.025
              },
              metric=['accuracy'])


# train
tensorboard_visualiser = TensorBoard()
checkpointer = ModelCheckpoint(filepath="./weights.hdf5", monitor="val_loss", verbose=1, save_best_only=True)
# monitor for accuracy not increasing anymore
early_stop = EarlyStopping(verbose=1, mode="max", patience=2, monitor="val_loss", min_delta=0.001)
class_weights = sklearn.utils.compute_class_weight(
                                                   class_weight='balanced',
                                                   classes=np.array(IDX_TO_CHAR.keys(), dtype=np.int64),
                                                   y=y_train.flatten())
class_weights = { idx: weight for idx, weight in enumerate(class_weights) }

history = model.fit(x_train, {name:y_train[:,idx] for idx, name in enumerate(loss_names)},
          epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE,
          verbose=1,
          shuffle=True,
          validation_split=0.1,
          class_weight=class_weights,
          callbacks=[tensorboard_visualiser, checkpointer])

# freeze layers for first two and last two characters
for i in [0, 1, 5, 6]:
    model.get_layer("output" + str(i)).trainable = False



# evaluate
score = model.evaluate(x_test, {name:y_test[:,idx] for idx, name in enumerate(loss_names)}, verbose=1)
logging.debug('Test loss: ', score[0])
logging.debug('Test accuracy: ', score[1])

y_pred = model.predict(x_test, verbose=1)
y_pred = np.argmax(y_pred, axis=2)
y_pred = np.transpose(y_pred)
num_correct = 0
for pred, test in zip(y_pred, y_test):
    if np.min(pred == test):
        num_correct += 1
        
accuracy = num_correct / float(len(y_pred))
logging.debug("No. correct: {}, Accuracy: {}".format(num_correct, accuracy))

logging.debug("Results:")
for pred, test in zip(y_pred, y_test):
    pred_str = "".join([IDX_TO_CHAR[i] for i in pred])
    test_str = "".join([IDX_TO_CHAR[i] for i in test])
    if pred_str.lower() != test_str.lower():
        logging.debug("Actual: {} Pred: {} [Wrong]".format(test_str, pred_str))
        

model.save('latest_model_150517.h5')