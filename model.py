import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D
from scipy import ndimage

# 1) Read lines from driving_log.csv file

lines = []

print('Opening .csv file...')

with open('/home/workspace/CarND-Behavioral-Cloning-P3/recorded_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# hint: use the following line to delete column titles in sample data
# del lines[0]

# 2) Read images and corresponding angles based on file path

images = []
measurements = []

print('Reading images...')

for line in lines:
        # iterate through lines in csv file
        steering_center = float(line[3])
        correction = 0.2    # angle correction for left and right images
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        steering_angles = (steering_center, steering_left, steering_right)
        
        for pos in range(3):
            # iterate through left-center-right images
            source_path = line[pos]
            filename = source_path.split('/')[-1]
            current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/recorded_data/IMG/' + filename
            
            image = ndimage.imread(current_path)
            images.append(image)
            
            measurement = steering_angles[pos]
            measurements.append(measurement)
            
            # hint: use following lines for augmentation by horizontal flipping
            # image_flipped = np.fliplr(image)
            # images.append(image_flipped)
            # measurement_flipped = -measurement
            # measurements.append(measurement_flipped)
        
X_train = np.array(images)
y_train = np.array(measurements)

# 3) Neural network model

model = Sequential()

# Crop & normalize
model.add(Cropping2D(cropping=((100,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(45,320,3)))

# Convolutional neural network
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Flatten())

model.add(Dense(1164))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('elu'))

model.add(Dense(1))

# 4) Compile, train and save model
      
print('Compiling...')
model.compile(loss='mse', optimizer='adam')

print('Training model...')
model.fit(X_train, y_train, validation_split=0.05, shuffle=True, nb_epoch=10)

print('Saving model...')
model.save('model.h5')