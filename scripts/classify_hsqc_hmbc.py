from time import gmtime, strftime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from sklearn.model_selection import KFold


#This script needs the hmbc and hsqc spectrum from each compound. Either remove the hsqc spectra without hmbc, or add blank images for hmbc to make them match
train_datagen=ImageDataGenerator(rescale=1./255)
train_set_hmbc=train_datagen.flow_from_directory('../data/hmbc/train',target_size=(300,205),batch_size=1000,color_mode='grayscale',class_mode='categorical',shuffle=False)
train_set_hsqc=train_datagen.flow_from_directory('../data/hsqc/train',target_size=(300,205),batch_size=1000,color_mode='grayscale',class_mode='categorical',shuffle=False)

# Ensured the images are in order
# print(train_set_hmbc.filenames)
# print('\n\n')
# print(train_set_hsqc.filenames)

hmbc_imgs, hmbc_targets = train_set_hmbc.next()
hsqc_imgs, hsqc_targets = train_set_hsqc.next()

num_folds = [2, 3, 5, 10, 82]
epochs = 50
acc_per_fold = []
loss_per_fold = []

output_path = "../output/"
filename = str(strftime("%a%d%b%Y_at_%H%M", gmtime())) + ".txt"
f = open(output_path+filename, "w+")

try:
    for i in range(len(num_folds)):
        print('-----------------------------------')
        f.write('\n-----------------------------------')
        print("NUM_FOLD SET TO: " + str(num_folds[i]))
        f.write("\nNUM_FOLD SET TO: " + str(num_folds[i]))
        print('-----------------------------------')
        f.write('\n-----------------------------------')
        #cross validation
        kfold = KFold(n_splits=num_folds[i], shuffle=False)
        fold_no = 1
        for train, test in kfold.split(hmbc_imgs, hmbc_targets):
            #setting up the model
            hmbc_input = keras.Input(
                shape=(300, 205, 1), name="hmbc"
            ) 
            hsqc_input = keras.Input(
                shape=(300, 205, 1), name="hsqc"
            )  

            #the hmbc "column"
            hmbc_conv1 = layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 205, 1))(hmbc_input)
            hmbc_softmax1 = layers.MaxPooling2D(2,2)(hmbc_conv1)
            hmbc_conv2 = layers.Conv2D(64, (3,3), activation='relu')(hmbc_softmax1)
            hmbc_softmax2 = layers.MaxPooling2D(2,2)(hmbc_conv2)
            hmbc_conv3 = layers.Conv2D(64, (3,3), activation='relu')(hmbc_softmax2)
            hmbc_softmax3 = layers.MaxPooling2D(2,2)(hmbc_conv3)
            hmbc_conv4 = layers.Conv2D(128, 3, activation='relu')(hmbc_softmax3)
            hmbc_flatten = layers.Flatten()(hmbc_conv4)

            #the hsqc "column"
            hsqc_conv1 = layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 205, 1))(hsqc_input)
            hsqc_softmax1 = layers.MaxPooling2D(2,2)(hsqc_conv1)
            hsqc_conv2 = layers.Conv2D(64, (3,3), activation='relu')(hsqc_softmax1)
            hsqc_softmax2 = layers.MaxPooling2D(2,2)(hsqc_conv2)
            hsqc_conv3 = layers.Conv2D(64, (3,3), activation='relu')(hsqc_softmax2)
            hsqc_softmax3 = layers.MaxPooling2D(2,2)(hsqc_conv3)
            hsqc_conv4 = layers.Conv2D(128, 3, activation='relu')(hsqc_softmax3)
            hsqc_flatten = layers.Flatten()(hsqc_conv4)

            #concetenate
            concatted = layers.Concatenate()([hsqc_flatten, hmbc_flatten])

            #the output
            dense = layers.Dense(64, activation='relu')(concatted)
            output = layers.Dense(3, activation='softmax', name='structure')(dense)

            model = keras.Model(
                inputs=[hmbc_input, hsqc_input],
                outputs=[output],
            )
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print('model compiled')
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            f.write(f'\nTraining for fold {fold_no} ...')
            
            #Ensuring that our data match
            if(hmbc_targets.all() == hsqc_targets.all()):
                model.fit(x=[hmbc_imgs[train], hsqc_imgs[train]], y=hmbc_targets[train],  epochs=epochs)
                # Generate generalization metrics
                scores = model.evaluate(x=[hmbc_imgs[test], hsqc_imgs[test]], y=hmbc_targets[test], verbose=0)
                print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
                f.write(f'\nScore for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])

                # Increase fold number
                fold_no = fold_no + 1


        print("\n\nOverall accuracy: " + str(np.average(acc_per_fold)))
        print("Overall loss: " + str(np.average(loss_per_fold)))

        f.write("\n\nOverall accuracy: " + str(np.average(acc_per_fold)))
        f.write("\nOverall loss: " + str(np.average(loss_per_fold)))
        f.write('\n\n===========================================\n')

except Exception as e:
    print(type(e))    # the exception instance
    print(e.args)     # arguments stored in .args
    print(e)          # __str__ allows args to be printed directly,
finally:
    f.close()