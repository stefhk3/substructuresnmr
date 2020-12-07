import argparse, collections, operator, os
from time import gmtime, strftime

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models


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

    print(labeled_prediction)

    sorted_p = collections.OrderedDict(sorted(labeled_prediction.items(), key=operator.itemgetter(1), reverse=True))

    # print(sorted_predictions)
    first_item = sorted_p.popitem(last=False)
    second_item = sorted_p.popitem(last=False)
    third_item = sorted_p.popitem(last=False)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filepath', help='path of the model with h5 extension')
    args = parser.parse_args()

    network = models.load_model(args.filepath)
    
    test_the_model()
