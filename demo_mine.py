# import the necessary packages
import json
import os
import random
import pandas
import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io

from utils import load_model


def get_class_name(my_class_id):
    column_titles = ['class_id', 'class_name']
    data = pandas.read_csv('devkit\cars_meta.csv', encoding="utf-8")
    dict_of_list = {}
    for column in column_titles:
        dict_of_list[column] = data[column].tolist()
    for class_id, class_name in zip(dict_of_list['class_id'], dict_of_list['class_name']):
        if my_class_id == class_id:
            return class_name


if __name__ == '__main__':
    img_width, img_height = 224, 224
    model = load_model()

    # cars_meta = scipy.io.loadmat('devkit/cars_meta')
    # class_names = cars_meta['class_names']  # shape=(1, 196)
    # class_names = np.transpose(class_names)

    test_path = 'demoimg/'
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]

    # num_samples = 20
    # samples = random.sample(test_images, num_samples)
    results = []
    header = "{:<50} {:<50} {}".format('Image Name', 'Predicted', 'Probability') + '\n' + "{:<50} {:<50} {}".format('-' * 50, '-' * 50, '-' * 15)
    print(header)
    for image in test_images:
        filename = os.path.join(test_path, image)
        # print('Start processing image: {}'.format(filename))
        bgr_img = cv.imread(filename)
        bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        # text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
        results.append({'label': get_class_name(class_id), 'prob': '{:.4}'.format(prob)})
        # cv.imwrite('images/{}_out.png'.format(image), bgr_img)
        resultstr = "{:<50} {:<50} {}".format(image, get_class_name(class_id), '{:.4}'.format(prob))
        print(resultstr)

    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

    K.clear_session()

