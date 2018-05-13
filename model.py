import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random import shuffle

samples  = []
with open('./recorded_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    image = cv2.imread(source_path)
                    images.append(image)
                    measurement = float(line[3])
                    measurements.append(measurement)

                augmented_images, augmented_measurements = [], []
                for image, measurement in zip(images, measurements):
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_measurements.append(measurement * -1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield X_train, y_train

# compile and train the model using the generator function
train_generator      = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

# Dimensions of the images
ch, row, col = 3, 160, 320

# Initialization
model = Sequential()
# Crop the top image which unused
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
# Normalized the image
model.add(Cropping2D(cropping=((60, 25), (0, 0))))

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