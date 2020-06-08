import csv
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, MaxPool2D, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
# from keras.applications.inception_resnet_v2 import InceptionResNetV2

# creat a generator for keras.fit_generator to provide finest-grained control over data
def generator(samples, b_size=128):
    correction = 0.25
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, b_size):
            batch_samples = samples[offset: offset + b_size]
            counter = 0
            images = []
            measurements = []
            for line in batch_samples:
                # print(line[3])
                if line[3] == 0:
                    counter += 1
                    # ignore some portion of images which has 0 steering angle to avoid biased training data
                    if counter % 5 == 0:
                        continue
                # Each timeframe will provide 6 images to the dataset using data augumentation
                current_steerings = [float(line[3]), float(line[3]) + correction, float(line[3]) - correction]
                for i in range(3):
                    name = '/opt/carnd_p3/data/IMG/' + line[i].split('/')[-1]
                    img = ndimage.imread(name)
                    images.append(img)
                    measurements.append(current_steerings[i])
                    # Flip over the image
                    image_flipped = np.fliplr(img)
                    images.append(image_flipped)
                    # Reverse the controlvalue
                    measurements.append(-current_steerings[i])

            X_train = np.array(images)
            Y_train = np.array(measurements)
            # every time the generator is called, a batch_size of data will be drawn
            yield shuffle(X_train, Y_train)


lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for c, line in enumerate(reader):
        if c == 0:
            continue
        lines.append(line)

# use sklearn to spilit data into training and validation set with portion 80/20
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

batch_size = 32

# Generator initialization
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# Creat callback so that the best model will be saved automatically
checkpoint = ModelCheckpoint('/home/workspace/CarND-Behavioral-Cloning-P3/model.h5', monitor='val_loss', mode = 'min', save_best_only=True)

# Also, it is possible to use transfer learning

# inception = InceptionResNetV2(include_top=False, weights = "imagenet")
# for layer in inception.layers:
#     layer.trainable = False
# model0 = Model(inputs = inception.input, outputs = inception.get_layer('activation_2').output)


model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# model.add(model0)
model.add(Convolution2D(32, 5, activation='relu'))
model.add(MaxPool2D(3, 3))
model.add(Convolution2D(64, 5, activation='relu'))
model.add(MaxPool2D(3, 3))
model.add(Convolution2D(128, 5, activation='relu'))
model.add(MaxPool2D(3, 3))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

history = model.fit_generator(train_generator, steps_per_epoch = np.ceil(len(train_samples)/batch_size), validation_data = validation_generator, validation_steps= np.ceil(len(validation_samples)/batch_size), epochs = 20, verbose = 2,
 callbacks=[checkpoint]
)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
