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


def collect_left_camera_images_with_steering_angles(data_array, image_data, center_angle, line):
    left_name = image_data + line[1].split('/')[-1]
    left_angle = center_angle + 0.35
    left_line = [left_name, left_angle]
    data_array.append(left_line)
    return data_array


def collect_right_camera_images_with_steering_angles(data_array, image_data, center_angle, line):
    right_name = image_data + line[2].split('/')[-1]
    right_angle = center_angle - 0.35
    right_line = [right_name, right_angle]
    data_array.append(right_line)
    return data_array


def collect_and_duplicate_mid_camera_images_with_steering_angles(data_array, center_angle, next_line):
    if center_angle < -0.15:
        for i in range(10):
            data_array.append(next_line)
    if center_angle > 0.15:
        for i in range(4):
            data_array.append(next_line)
    return data_array


def read_entries(data_array, driving_log_data, image_data):
    with open(driving_log_data) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                center_angle = float(line[3])
                name = image_data + line[0].split('/')[-1]
                next_line = [name, center_angle]

                lef_data = collect_left_camera_images_with_steering_angles(data_array, image_data, center_angle, line)
                right_data = collect_right_camera_images_with_steering_angles(lef_data, image_data, center_angle, line)
                right_data.append(next_line)
                total_data = collect_and_duplicate_mid_camera_images_with_steering_angles(right_data, center_angle, next_line)
            except ValueError:
                print("Unable to read entries")
        return total_data


def plot_model_accuracy(model_history):
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('The Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_model_loss(model_history):
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('The Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


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

    # Add 5 Fully Connected Layers, 4 Activation function and 1 Dropout layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
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

data = read_entries(data, my_data_driving_log, my_data_image)

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
# input shape
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

'''
Plot Model Accuracy
'''
plot_model_accuracy(history)

'''
Plot Model Lost
'''
plot_model_loss(history)
