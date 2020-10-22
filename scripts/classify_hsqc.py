from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras import layers
import numpy as np
#import matplotlib.pyplot as plt

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

#test_datagen=ImageDataGenerator(rescale=1./255)
train_datagen=ImageDataGenerator(rescale=1./255)
#test_set=test_datagen.flow_from_directory('test',target_size=(1024,1024),batch_size=32,class_mode='binary')
train_set=train_datagen.flow_from_directory('../data/hsqc/train',target_size=(300,205),batch_size=8,color_mode='grayscale',class_mode='categorical')
#build network
network=models.Sequential()
network.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 205, 1)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64, (3,3), activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64, (3,3), activation='relu'))
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
img_path = '../data/hsqc/test/fattyacids/bmse000502_1H_13C_HSQC.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/fattyacids/502_60.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/fattyacids/502_61.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/fattyacids/502_60_61.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/fattyacids/502_60_61_70.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/fattyacids/502_60_61_70_491.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/fattyacids/502_60_61_70_491_544.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/indol/bmse000364_1H_13C_HSQC.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/indol/364_60.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/indol/364_60_61.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/indol/364_60_61_70.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/indol/364_60_61_70_491.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/indol/364_60_61_70_491_544.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/steroids/bmse000489_1H_13C_HSQC.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/steroids/489_60.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/steroids/489_60_61.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/steroids/489_60_61_70.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/steroids/489_60_61_70_491.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
img_path = '../data/hsqc/test/steroids/489_60_61_70_491_544.jpg'
new_image = load_image(img_path, False)
pred = network.predict(new_image)
print(pred)
