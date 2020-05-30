import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

def reformat(df):
    labels = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    reformatted_data = []
    for i in range(len(data)):
        narray = data.iloc[i, :].to_numpy().ravel()
        narray = narray.reshape(int(math.sqrt(len(narray))), int(math.sqrt(len(narray))), 1)
        reformatted_data.append(narray)

    return labels.to_numpy().ravel(), np.array(reformatted_data)

def create_augmented_data(data, labels):
    datagen = ImageDataGenerator(
        shear_range=0.5,
        rotation_range=90,
        zoom_range=(0.9, 1.1),
        horizontal_flip=False,
        vertical_flip=False)

    datagen.fit(data)

    return datagen.flow(x=data,
                        y=labels,
                        batch_size=32)

def one_hot_encode(labels):
    encoded = []
    max_val = labels.max()
    min_val = labels.min()
    for data in labels:
        r = np.zeros(max_val - min_val + 1)
        r[data - min_val] = 1
        encoded.append(r)

    return np.array(encoded)

def one_hot_to_index(labels):
    return np.argmax(labels, axis=1)

def calc_acc(classes, model, X, y, title):
    correct = 0
    incorrect = 0

    x_result = model.predict(X, verbose=0)

    class_correct = [0] * len(classes)
    class_incorrect = [0] * len(classes)

    for i in range(len(X)):
        act = y[i]
        res = x_result[i]

        actual_label = int(np.argmax(act))
        pred_label = int(np.argmax(res))

        if pred_label == actual_label:
            class_correct[actual_label] += 1
            correct += 1
        else:
            class_incorrect[actual_label] += 1
            incorrect += 1

    acc = float(correct) / float(correct + incorrect)

    result_string = ""
    result_string += "Current Network " + title + " Accuracy: %.3f \n\n" % (acc)
    result_string += "Current Network " + title + " Class Accuracies:\n"
    for i in range(len(classes)):
        tot = float(class_correct[i] + class_incorrect[i])
        class_acc = -1
        if (tot > 0):
            class_acc = float(class_correct[i]) / tot

        result_string += "\t%s: %.3f\n" % (classes[i], class_acc)

    if print_accuracy_to_file:
        if not os.path.isdir("result"):
            os.mkdir("result")
        f = open("result\\"+title.lower()+"_result.txt","w+")
        f.write(result_string)
        f.close()
        print("Printing "+title+" accuracy is done!")
    else:
        print(result_string)

class AccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_train_end(self, logs=None):
        x_data, y_data = self.test_data

        calc_acc(classes, self.model, x_data, y_data, "Train")

# Settings

set_already_trained_model = True
plot_loss = False
print_accuracy_to_file = False

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
           "Ankle boot"]

# Data Preprocess
raw_train = pd.read_csv("data\\fashion-mnist_train.csv")
raw_test = pd.read_csv("data\\fashion-mnist_test.csv")

train_labels, train = reformat(raw_train)
test_labels, test = reformat(raw_test)

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

# Model creation

classifier = Sequential()

classifier.add(Convolution2D(32, kernel_size=3, input_shape=(28, 28, 1), activation='relu', padding='same'))

classifier.add(Dropout(0.25))

classifier.add(Convolution2D(64, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(2))

classifier.add(Flatten())

classifier.add(Dense(units=256, activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=10, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Setting callbacks

accuracy_callback = AccuracyCallback((train, train_labels))
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', save_best_only=True)

# Training/Loading
if set_already_trained_model:
    classifier.load_weights('weights.hdf5')
else:
    history = classifier.fit(train,
                             train_labels,
                             epochs=5,
                             shuffle=True,
                             validation_split=1 / 6,
                             callbacks=[accuracy_callback, checkpoint])

# Plotting

if plot_loss and not set_already_trained_model:
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Validation")
    plt.legend()
    plt.show()

# Reporting

calc_acc(classes, classifier, test, test_labels, "Test")
