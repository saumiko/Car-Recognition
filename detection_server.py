from flask import Flask
from flask import request
import json
import pandas
import cv2 as cv
import numpy as np
from resnet_152 import resnet152_model
import tensorflow as tf


app = Flask(__name__)


def load_model():
    global det_model
    global graph
    model_weights_path = 'models/model.96-0.89.hdf5'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    det_model = resnet152_model(img_height, img_width, num_channels, num_classes)
    det_model.load_weights(model_weights_path, by_name=True)
    graph = tf.get_default_graph()


def load_classes():
    global car_classes
    column_titles = ['class_id', 'class_name']
    data = pandas.read_csv('cars_meta.csv', encoding="utf-8")
    dict_of_list = {}
    for column in column_titles:
        dict_of_list[column] = data[column].tolist()
    car_classes = [None] * (max(dict_of_list['class_id'])+1)
    for class_id, class_name in zip(dict_of_list['class_id'], dict_of_list['class_name']):
        car_classes[int(class_id)] = class_name


det_model = None
graph = None
load_model()
car_classes = None
load_classes()


def get_class_name(my_class_id):
    return car_classes[int(my_class_id)]


def get_vehicle_model(img):
    img_width, img_height = 224, 224
    bgr_img = cv.imdecode(np.fromstring(img.read(), np.uint8), cv.IMREAD_UNCHANGED)
    bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
    with graph.as_default():
        preds = det_model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        result = {
            "model": get_class_name(class_id),
            "class_id": int(class_id),
            "probability": float(prob),
            'filename': img.filename
        }
        return result


@app.route("/recognize-car", methods=['POST'])
def detect():
    return json.dumps(get_vehicle_model(request.files['file']))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5007, debug=False)
