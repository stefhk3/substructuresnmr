from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from sklearn.model_selection import KFold


#This script needs the hmbc and hsqc spectrum from each compound. Either remove the hsqc spectra without hmbc, or add blank images for hmbc to make them match

num_folds = 10
acc_per_fold = []
loss_per_fold = []


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

class JoinedGenerator(keras.utils.Sequence):
    def __init__(self, generator1, generator2):
        self.generator1 = generator1
        self.generator2 = generator2 

    def __len__(self):
        return len(self.generator1)

    def __getitem__(self, i):
        x1, y1 = self.generator1[i]
        x2, y2 = self.generator2[i]
        return [x1, x2], y1

    def on_epoch_end(self):
        self.generator1.on_epoch_end()
        self.generator2.on_epoch_end()


train_datagen=ImageDataGenerator(rescale=1./255)
train_set_hmbc=train_datagen.flow_from_directory('../data/hmbc/train',target_size=(300,205),batch_size=8,color_mode='grayscale',class_mode='categorical',shuffle=False)
train_set_hsqc=train_datagen.flow_from_directory('../data/hsqc/train',target_size=(300,205),batch_size=8,color_mode='grayscale',class_mode='categorical',shuffle=False)
training_generator = JoinedGenerator(train_set_hmbc, train_set_hsqc)


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

#cross validation

kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
for train, test in kfold.split(training_generator):
    model = keras.Model(
        inputs=[hmbc_input, hsqc_input],
        outputs=[output],
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('model compiled')
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    model.fit_generator(generator=training_generator,  epochs=50)
        # Generate generalization metrics
    scores = model.evaluate(x[test], y[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1


print("\n\n Overall accuracy: " + str(np.average(acc_per_fold)))
print("Overall loss: " + str(np.average(loss_per_fold)))
