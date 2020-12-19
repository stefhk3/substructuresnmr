# when using the last push of the code in the master branch, there are flags to be set.

# save_best_in_fold         is the flag to toggle when the models need to be saved or not

# save_models_num_folds      is the number of folds 

# minimum_accuracy = "75".   is the threshold for which models are required to be saved. Currently this is set to 75.

# The saved models' names include their accuracies, hence the undesirable models can be removed from the directory if wanted.


from time import gmtime, strftime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras

import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold


#This script needs the hmbc and hsqc spectrum from each compound. Either remove the hsqc spectra without hmbc, or add blank images for hmbc to make them match
train_datagen=ImageDataGenerator(rescale=1./255)
train_set_hmbc=train_datagen.flow_from_directory('../data/hmbc/train',target_size=(300,205),batch_size=1000,color_mode='grayscale',class_mode='categorical',shuffle=False)
train_set_hsqc=train_datagen.flow_from_directory('../data/hsqc/train',target_size=(300,205),batch_size=1000,color_mode='grayscale',class_mode='categorical',shuffle=False)

hmbc_imgs, hmbc_targets = train_set_hmbc.next()
hsqc_imgs, hsqc_targets = train_set_hsqc.next()

num_folds = [2, 3, 5, 10, 82]
save_models_in_fold = True
save_models_num_folds = 10
minimum_accuracy = "75"

epochs = 50
acc_per_fold = []
loss_per_fold = []

output_path = "../output/"
filename = str(strftime("%a%d%b%Y_at_%H%M", gmtime())) + ".txt"
f = open(output_path+filename, "w+")

def turn_categorical_to_one_d_array(cat_array):
    one_d_array = []
    for i in range(len(cat_array)):
        for item_no in range(len(cat_array[i])):
            if cat_array[i][item_no] == 1:
                one_d_array.append(item_no)
    return one_d_array

def model_setup():
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

    return model

def kfolds_train_and_test(num_folds, flag_save_best):
    #for saving models that achieved above 50% accuracy
    folds = []
    global acc_per_fold, loss_per_fold

    if not(isinstance(num_folds, int)) and flag_save_best==True:
        raise Exception('Arrays of fold numbers are not accepted when the best model is to be saved')
    elif isinstance(num_folds, int) and flag_save_best==True:
        folds.append(num_folds)
    elif not(isinstance(num_folds, int)) and flag_save_best!=True:
        folds = num_folds

    print('out')
    print(folds[0])

    for i in range(len(folds)):
        print('-----------------------------------')
        f.write('\n-----------------------------------')
        print("NUM_FOLD SET TO: " + str(folds[i]))
        f.write("\nNUM_FOLD SET TO: " + str(folds[i]))
        print('-----------------------------------')
        f.write('\n-----------------------------------')
        #cross validation
        kfold = StratifiedKFold(n_splits=folds[i], shuffle=False)
        fold_no = 1
        encoded_hmbc_targets = turn_categorical_to_one_d_array(hmbc_targets)
        
        for train, test in kfold.split(hmbc_imgs, encoded_hmbc_targets):
            model = model_setup()
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

                if(flag_save_best==True):
                    print(acc_per_fold[-1])

                    if acc_per_fold[-1] >= float(minimum_accuracy):
                        # FOR SAVING THE TRAINED MODEL #
                        current_datetime = str(strftime("%a%d%b%Y_at_%H%M", gmtime()))
                        # file_name = model_accuracy + "accuracy on " + current_datetime + ".h5"
                        file_name = current_datetime + "_acc_"+ str(scores[1]*100) + ".h5"
                        try:
                            model.save("../models/" + file_name)
                            print("\n The model is successfully saved in models folder with the name: " + file_name)
                        except(Exception):
                            print("\n An error occured while saving the model: " + sys.exc_info()[0])
                            raise

                # Increase fold number
                fold_no = fold_no + 1
                
        print("\n\nOverall accuracy: " + str(np.average(acc_per_fold)))
        print("Overall loss: " + str(np.average(loss_per_fold)))

        f.write("\n\nOverall accuracy: " + str(np.average(acc_per_fold)))
        f.write("\nOverall loss: " + str(np.average(loss_per_fold)))
        f.write('\n\n===========================================\n')

        #resetting the arrays
        acc_per_fold = []
        loss_per_fold = []

try:
    if save_models_in_fold != True:
        for num_fold in range(len(num_folds)):
            kfolds_train_and_test(num_fold, False)
    else:
        kfolds_train_and_test(save_models_num_folds, True)

except Exception as e:
    print(type(e))    # the exception instance
    print(e.args)     # arguments stored in .args
    print(e)          # __str__ allows args to be printed directly,
finally:
    f.close()
