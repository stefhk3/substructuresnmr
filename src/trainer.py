import collections, operator, os
from time import gmtime, strftime

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras import layers


#import matplotlib.pyplot as plt

#test_datagen=ImageDataGenerator(rescale=1./255)
train_datagen=ImageDataGenerator(rescale=1./255)
#test_set=test_datagen.flow_from_directory('test',target_size=(1024,1024),batch_size=32,class_mode='binary')
train_set=train_datagen.flow_from_directory('../data/hsqc/train',target_size=(300,205),batch_size=8,color_mode='grayscale',class_mode='categorical')
#build network
network=models.Sequential()
network.add(layers.Conv2D(32, 3, activation='relu', input_shape=(300, 205, 1)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64, 3, activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64, 3, activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(128, 3, activation='relu'))
network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(3, activation='softmax'))
#The default learning rate is 0.01
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#perform training
network.fit(train_set, epochs=50)
#test using test data
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# model_accuracy = str(history.history['accuracy'])
current_datetime = str(strftime("%a%d%b%Y_at_%H%M", gmtime()))
# file_name = model_accuracy + "accuracy on " + current_datetime + ".h5"
file_name = current_datetime + ".h5"

try:
    network.save("../models/" + file_name)
    print("\n The model is successfully saved in models folder with the name: " + file_name)
except(Exception):
    print("\n An error occured while saving the model: " + sys.exc_info()[0])
    raise


