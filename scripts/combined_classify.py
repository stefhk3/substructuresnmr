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

epochs = 50

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

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(300, 205),color_mode='grayscale')
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    #if show:
    #    plt.imshow(img_tensor[0])                           
    #    plt.axis('off')
    #    plt.show()

    return img_tensor

def classify():
    #for saving models that achieved above 50% accuracy
    encoded_hmbc_targets = turn_categorical_to_one_d_array(hmbc_targets)
        
    model = model_setup()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('model compiled')
    #Ensuring that our data match
    if(hmbc_targets.all() == hsqc_targets.all()):
        model.fit(x=[hmbc_imgs, hsqc_imgs], y=hmbc_targets,  epochs=epochs)

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    img_path = '../data/hsqc/test/fattyacids/bmse000643_1H_13C_HSQC.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/fattyacids/bmse000643_1H_13C_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/fattyacids/643_60.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/fattyacids/643_60_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/fattyacids/643_60_61.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/fattyacids/643_60_61_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/fattyacids/643_60_61_70.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/fattyacids/643_60_61_72_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/fattyacids/643_60_61_70_491.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/fattyacids/643_60_61_72_499_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/fattyacids/643_60_61_70_491_544.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/fattyacids/643_60_61_72_499_544_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    #img_path = '../data/hsqc/test/indol/bmse000364_1H_13C_HSQC.jpg'
    img_path = '../data/hsqc/train/indol/bmse000516_1H_13C_HSQC.jpg'
    new_image = load_image(img_path, False)
    #img_path2 = '../data/hmbc/test/indol/bmse000364_1H_13C_HMBC.jpg'
    img_path2 = '../data/hmbc/train/indol/bmse000516_1H_13C_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/indol/364_60.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/indol/364_60_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/indol/364_60_61.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/indol/364_60_61_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/indol/364_60_61_70.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/indol/364_60_61_72_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/indol/364_60_61_70_491.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/indol/364_60_61_72_499_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/indol/364_60_61_70_491_544.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/indol/364_60_61_72_499_544_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/steroids/bmse000489_1H_13C_HSQC.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/steroids/bmse000489_1H_13C_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/steroids/489_60.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/steroids/489_60_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/steroids/489_60_61.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/steroids/489_60_61_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/steroids/489_60_61_70.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/steroids/489_60_61_72_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/steroids/489_60_61_70_491.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/steroids/489_60_61_72_499_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)
    img_path = '../data/hsqc/test/steroids/489_60_61_70_491_544.jpg'
    new_image = load_image(img_path, False)
    img_path2 = '../data/hmbc/test/steroids/489_60_61_72_499_544_HMBC.jpg'
    new_image2 = load_image(img_path2, False)
    pred = model.predict([new_image, new_image2])
    print(pred)

try:
    classify()

except Exception as e:
    print(type(e))    # the exception instance
    print(e.args)     # arguments stored in .args
    print(e)          # __str__ allows args to be printed directly,
finally:
    f.close()
