# -*- coding: utf-8 -*-

import tarfile
import scipy.io
import numpy as np
import os
import cv2 as cv
import shutil
import random
import pandas
from console_progressbar import ProgressBar


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_train_data(fnames, labels, bboxes):
    src_folder = 'cars_train'
    num_samples = len(fnames)

    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    train_indexes = random.sample(range(num_samples), num_train)

    pb = ProgressBar(total=100, prefix='Save train data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        label = labels[i]
        (x1, y1, x2, y2) = bboxes[i]

        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print("{} -> {}".format(fname, label))
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        if i in train_indexes:
            dst_folder = 'data/train'
        else:
            dst_folder = 'data/valid'

        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, fname)

        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)


def save_test_data(fnames, bboxes):
    src_folder = 'cars_test'
    dst_folder = 'data/test'
    num_samples = len(fnames)

    pb = ProgressBar(total=100, prefix='Save test data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        (x1, y1, x2, y2) = bboxes[i]
        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print(fname)
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        dst_path = os.path.join(dst_folder, fname)
        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)


def process_train_data():
    column_titles = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
    data = pandas.read_csv('devkit\cars_train_annos.csv', encoding="utf-8")
    dict_of_list = {}
    for column in column_titles:
        dict_of_list[column] = data[column].tolist()

    fnames = []
    bboxes = []
    labels = []

    for bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id, fname in zip(dict_of_list.get('bbox_x1'),
                                                                   dict_of_list.get('bbox_y1'),
                                                                   dict_of_list.get('bbox_x2'),
                                                                   dict_of_list.get('bbox_y2'),
                                                                   dict_of_list.get('class'),
                                                                   dict_of_list.get('fname')):
        # print(bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id, fname)
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        labels.append('%04d' % (class_id),)
        fnames.append(fname)
    save_train_data(fnames, labels, bboxes)


def process_test_data():
    column_titles = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
    data = pandas.read_csv('devkit\cars_test_annos.csv', encoding="utf-8")
    dict_of_list = {}
    for column in column_titles:
        dict_of_list[column] = data[column].tolist()

    fnames = []
    bboxes = []

    for bbox_x1, bbox_y1, bbox_x2, bbox_y2, fname in zip(dict_of_list.get('bbox_x1'),
                                                                   dict_of_list.get('bbox_y1'),
                                                                   dict_of_list.get('bbox_x2'),
                                                                   dict_of_list.get('bbox_y2'),
                                                                   dict_of_list.get('fname')):
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        fnames.append(fname)

    save_test_data(fnames, bboxes)


if __name__ == '__main__':
    # parameters
    img_width, img_height = 224, 224

    print('Extracting cars_train.tgz...')
    if not os.path.exists('cars_train'):
        with tarfile.open('cars_train.tgz', "r:gz") as tar:
            tar.extractall()
    print('Extracting cars_test.tgz...')
    if not os.path.exists('cars_test'):
        with tarfile.open('cars_test.tgz', "r:gz") as tar:
            tar.extractall()
    print('Extracting car_devkit.tgz...')
    if not os.path.exists('devkit'):
        with tarfile.open('car_devkit.tgz', "r:gz") as tar:
            tar.extractall()

    ensure_folder('data/train')
    ensure_folder('data/valid')
    ensure_folder('data/test')

    process_train_data()
    process_test_data()

    # clean up
    shutil.rmtree('cars_train')
    shutil.rmtree('cars_test')
    # shutil.rmtree('devkit')
