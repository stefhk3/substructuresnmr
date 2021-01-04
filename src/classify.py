import argparse, collections, operator, os
from time import gmtime, strftime

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models

combined = False

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

def print_conclusion(prediction):
    #0: fattyacids 1:indols 2:steroids
    labels = ['fattyacid', 'indole', 'steroid']
    labeled_prediction = {}
    for x in range(3):
        labeled_prediction[labels[x]] = round(prediction[0][x], 3)

    sorted_p = collections.OrderedDict(sorted(labeled_prediction.items(), key=operator.itemgetter(1), reverse=True))

    # print(sorted_predictions)
    first_item = sorted_p.popitem(last=False)
    second_item = sorted_p.popitem(last=False)
    third_item = sorted_p.popitem(last=False)

    if combined:
        print('\n\n' + str(first_item[0]) + ' ' + 'with ' + str(first_item[1]*100) + '%% confidence')
    else:
        if (
                (str(first_item[1])[:3] == '0.3') 
                and 
                (str(second_item[1])[:3] == '0.3') 
                and 
                (str(third_item[1])[:3] == '0.3')
            ):
            print("equal mixes of all")
        elif (
                (float(str(first_item[1])[:3]) >= 0.4) 
                and 
                (float(str(second_item[1])[:3]) >= 0.4)
            ):
            print("equal mix of " + first_item[0] + " and " + second_item[0])
        else:
            print("dominant mix:  " + first_item[0])
            if (second_item[1] > 0.2):
                print("minor mix:  " + second_item[0])
            else:
                print("negligent mix: " + second_item[0])
            if (third_item[1] < 0.2):
                print("negligent mix:  " + third_item[0])
            else:
                print("minor mix:  " + third_item[0])

def test_the_model():
    test_path = '../data/hsqc/test/'
    folders = os.listdir(test_path)
    for folder in folders:
        if str(folder) != ".DS_Store":
            print(str(folder) + '\n')
            image_names = os.listdir(test_path + str(folder))
            for img in image_names:
                if str(img) != ".DS_Store":
                    loaded_img = load_image(test_path + '/' + str(folder) + '/' + img)
                    print_conclusion(network.predict(loaded_img))
        print('\n\n\n')

def model_classification(network, imgs):
    prediction = network.predict(imgs)
    print_conclusion(prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and run an h5 model.')
    parser.add_argument('modelpath', help='path of the model with h5 extension')
    parser.add_argument('imgpaths', nargs="+", help='path of the images')
    args = parser.parse_args()

    imgs = []

    for path in args.imgpaths:
        img = load_image(path)
        imgs.append(img)
    
    network = models.load_model(args.modelpath)

    if len(imgs) == 1:
        model_classification(network, imgs[0])
    else:
        combined = True
        model_classification(network, imgs)