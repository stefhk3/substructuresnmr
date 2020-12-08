#One-out automated
import collections, operator, os, sys
from time import gmtime, strftime

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras import layers

from sklearn.model_selection import KFold
import numpy as np

train_path = "../data/hsqc/train/"
test_path = "../data/hsqc/test/"
#3 folders inside for each class
# folders = os.listdir(test_path)
dataset_to_train = []


num_folds = 5
epochs = 50
acc_per_fold = []
loss_per_fold = []


train_datagen=ImageDataGenerator(rescale=1./255)

data=train_datagen.flow_from_directory(directory=train_path,
                                            target_size=(300,205), batch_size=70,
                                            color_mode='grayscale',class_mode='categorical')
# test_set=train_datagen.flow_from_directory(directory=test_path,
#                                             target_size=(300,205), batch_size=8, 
#                                             color_mode='grayscale',class_mode='categorical')

x, y = data.next()
# x, y = data.next()

print(x.shape)
print(y.shape)

# inputs = np.concatenate((train_set, test_set), axis=0)
# inputs = train_set.concatenate(test_set)
# inputs = chain(train_set, test_set)
print("Defining the K-fold Cross Validator")
kfold = KFold(n_splits=num_folds, shuffle=True)
print(kfold.get_n_splits(x, y))

fold_no = 1
for train, test in kfold.split(x, y):
    print("#build network")
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

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    network.fit(x[train], y[train], epochs=epochs)
    #test using test data
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    ## FOR SAVING THE TRAINED MODEL #
    # current_datetime = str(strftime("%a%d%b%Y_at_%H%M", gmtime()))
    # # file_name = model_accuracy + "accuracy on " + current_datetime + ".h5"
    # file_name = current_datetime + ".h5"

    # try:
    #     network.save("../models/" + file_name)
    #     print("\n The model is successfully saved in models folder with the name: " + file_name)
    # except(Exception):
    #     print("\n An error occured while saving the model: " + sys.exc_info()[0])
    #     raise

    # Generate generalization metrics
    scores = network.evaluate(x[test], y[test], verbose=0)
    print(f'Score for fold {fold_no}: {network.metrics_names[0]} of {scores[0]}; {network.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
