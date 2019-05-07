from scipy.io import loadmat
import numpy as np
import csv


def process_test_data():
    cars_annos = loadmat('devkit/cars_test_annos.mat')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)
    header = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
    csvdat = [header]

    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        fname = annotation[0][4][0]
        row = [bbox_x1, bbox_y1, bbox_x2, bbox_y2, fname]
        csvdat.append(row)

    with open('devkit/cars_test_annos.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in csvdat:
            writer.writerow(row)
    f.close()


def process_train_data():
    cars_annos = loadmat('devkit/cars_train_annos.mat')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)
    header = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
    csvdat = [header]

    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)
    
    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        class_id = annotation[0][4][0][0]
        fname = annotation[0][5][0]
        row = [bbox_x1, bbox_y1, bbox_x2, bbox_y2,class_id, fname]
        csvdat.append(row)
    
    with open('devkit/cars_train_annos.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in csvdat:
            writer.writerow(row)
    f.close()

def process_classes():
    cars_meta = loadmat('devkit/cars_meta.mat')
    class_names = cars_meta['class_names']
    # print(class_names)
    classes = []
    class_names = np.transpose(class_names)
    for i in range(0, class_names.shape[0]):
        classes.append(class_names[i][0][0])

    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)
    with open('devkit/cars_meta.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        writer.writerow(classes)
    f.close()

process_test_data()
process_train_data()
process_classes()
