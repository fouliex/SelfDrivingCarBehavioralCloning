# Load Python Modules
import numpy as np
import sklearn
import csv
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split


def read_entries(data_array, driving_log_data, image_data):
    with open('data/myData/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                center_angle = float(line[3])
                name = 'data/myData/IMG/' + line[0].split('/')[-1]

                newline = []
                newline.append(name)
                newline.append(center_angle)

                # left camera
                leftname = 'data/myData/IMG/' + line[1].split('/')[-1]
                leftangle = center_angle + 0.35
                leftline = []
                leftline.append(leftname)
                leftline.append(leftangle)
                data.append(leftline)

                # right camera
                rightname = 'data/myData/IMG/' + line[2].split('/')[-1]
                rightangle = center_angle - 0.35
                rightline = []
                rightline.append(rightname)
                rightline.append(rightangle)
                data.append(rightline)

                data.append(newline)

                # more emphasis on large steering samples
                if center_angle < -0.15:
                    data.append(newline)
                    data.append(newline)
                    data.append(newline)
                    data.append(newline)
                    data.append(newline)
                    data.append(newline)
                    data.append(newline)
                    data.append(newline)
                    data.append(newline)
                if center_angle > 0.15:
                    data.append(newline)
                    data.append(newline)
            except ValueError:
                a = 1


# Use Generator so we don't have to load all images into memory
def generator(data, batch_size):
    num_samples = len(data)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                try:
                    center_angle = float(batch_sample[1])
                    name = batch_sample[0]
                    srcBGR = cv2.imread(name)
                    center_image = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
                    images.append(center_image)
                    angles.append(center_angle)
                except ValueError:
                    a = 1
                    print("Not a number in ", batch_sample[0], " Value: ", batch_sample[1])
                except Exception:
                    print("Image error ", batch_sample[0], " Value: ", batch_sample[1])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def create_model(row, col, ch):

    model = Sequential()

    # Crop out the top and bottom parts of the image
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(row, col, ch)))

    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))

    # Reduce the image size by half by using Max Pooling
    model.add(MaxPooling2D((2, 2)))

    # Add 5 Convolution Layers
    model.add(Convolution2D(24, (5, 5), strides=2))
    model.add(Convolution2D(36, (5, 5), strides=2))
    model.add(Convolution2D(48, (5, 5), strides=1))
    model.add(Convolution2D(64, (3, 3), strides=1))
    model.add(Convolution2D(64, (3, 3), strides=1))

    # Flatten
    model.add(Flatten())

    # Add  5 Fully Connected Layers, with 1 Dropout layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # Compile with Adam Optimizer
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


'''
Training data collected by Driving the car around the lap

'''
data = []
my_data_driving_log = 'data/myData/driving_log.csv'
my_data_image = 'data/myData/IMG/'

read_entries(data, my_data_driving_log, my_data_image)

'''
Split data into Training and validation
'''
train_samples, validation_samples = train_test_split(data, test_size=0.2)

print("***Train Sample Size***", len(train_samples))
print("***Validation Sample Size***", len(validation_samples))

'''
Compile the Model
'''
epoch = 10
batch_size = 128
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)
# Trimmed image format
ch, row, col = 3, 160, 320
model = create_model(row, col, ch)
print(model.summary())

'''
Train the Model
'''
history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples) / batch_size - 1,
                              validation_data=validation_generator,
                              validation_steps=len(validation_samples) / batch_size - 1, epochs=epoch)


'''
Save the Model
'''
model.save("model.h5")