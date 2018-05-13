import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from random import shuffle

samples  = []
with open('./recorded_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images, angles = [], []
            for batch_sample in batch_samples:
                    name = './data/IMG/' + batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)

                    center_image_flipped = cv2.flip(center_image, 1)
                    center_angle_flipped = -center_angle
                    images.append(center_image_flipped)
                    angles.append(center_angle_flipped)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator      = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Initialization
model = Sequential()
# Crop the top image which unused
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
# Normalized the image
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
# model.add(Convolution2D(6, 5, 5, subsample=(2,2), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, subsample=(2,2), activation='relu'))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch = len(train_samples),
                                     validation_data = validation_generator,
                                     nb_val_samples = len(validation_samples),
                                     nb_epoch=10)

# model.save('model.h5')
model.save('model_nvidia.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('loss.jpg')