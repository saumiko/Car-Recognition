import os
import shutil
import csv
from random import shuffle
from bs4 import BeautifulSoup

# Set % ratio
training = 60
validation = 20
test = 20

# input path
input_dir = 'newannos/'

output_dir = os.path.join(input_dir, 'output')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

class_names = []
annotations = []
annotations_by_class = {}


def percentage(percent, whole):
  return int((percent * whole) / 100)


# Parse all annotation files and prepare data
for filename in os.listdir(input_dir):
    if filename.endswith('.xml'):
        annotation_file = open(os.path.join(input_dir, filename), encoding='utf8').read()
        annotation_soup = BeautifulSoup(annotation_file, 'xml')
        filename = annotation_soup.find('filename').getText()
        objects = annotation_soup.find_all('object')
        for object in objects:
            class_name = object.find('name').getText()
            if class_name not in class_names:
                class_names.append(class_name)
            bbox = object.find('bndbox')
            bbox_x1 = bbox.find('xmin').getText()
            bbox_y1 = bbox.find('ymin').getText()
            bbox_x2 = bbox.find('xmax').getText()
            bbox_y2 = bbox.find('ymax').getText()
            annotation = {
                'bbox_x1' : bbox_x1,
                'bbox_y1': bbox_y1,
                'bbox_x2': bbox_x2,
                'bbox_y2': bbox_y2,
                'class': class_names.index(class_name),
                'fname': filename
            }
            annotations.append(annotation)




# prepare annotations by class
for class_name in class_names:
    class_annos = []
    for annotation in annotations:
        if annotation['class'] == class_names.index(class_name):
            class_annos.append(annotation)
    annotations_by_class[class_name] = class_annos



csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)


def init_prep_data():
    header = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
    csvdat = [header]
    dats = ['/cars_train_annos.csv', '/cars_val_annos.csv', '/cars_test_annos.csv']
    for datfile in dats:
        with open(output_dir + datfile, 'w') as f:
            writer = csv.writer(f, dialect='myDialect')
            for row in csvdat:
                writer.writerow(row)
        f.close()


def prep_data(annotations, ind):
    # Write full processed meta data in 60, 20, 20 ratio
    # Training
    csvdat = []
    ckpt = 0
    shuffle(annotations)
    for i in range(ckpt, percentage(training, len(annotations))):
        row = annotations[i]
        rowdat = [
            row['bbox_x1'],
            row['bbox_y1'],
            row['bbox_x2'],
            row['bbox_y2'],
            ind,
            row['fname']
        ]
        csvdat.append(rowdat)

    with open(output_dir + '/cars_train_annos.csv', 'a+') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in csvdat:
            writer.writerow(row)
    f.close()

    # Validation
    csvdat = []
    ckpt = percentage(training, len(annotations))
    for i in range(ckpt, ckpt + percentage(validation, len(annotations))):
        row = annotations[i]
        rowdat = [
            row['bbox_x1'],
            row['bbox_y1'],
            row['bbox_x2'],
            row['bbox_y2'],
            ind,
            row['fname']
        ]
        csvdat.append(rowdat)

    with open(output_dir + '/cars_val_annos.csv', 'a+') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in csvdat:
            writer.writerow(row)
    f.close()

    # Test
    csvdat = []
    ckpt = ckpt + percentage(validation, len(annotations))
    for i in range(ckpt, ckpt + percentage(test, len(annotations))):
        row = annotations[i]
        rowdat = [
            row['bbox_x1'],
            row['bbox_y1'],
            row['bbox_x2'],
            row['bbox_y2'],
            ind,
            row['fname']
        ]
        csvdat.append(rowdat)

    with open(output_dir + '/cars_test_annos.csv', 'a+') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in csvdat:
            writer.writerow(row)
    f.close()


class_names = []
init_prep_data()
for key, value in annotations_by_class.items():
    if len(value) > 40:
        class_names.append(key)
        prep_data(value, class_names.index(key))
        # print(key, len(value))

# Write Meta Data
header = ['class_id', 'class_name']
csvdat = [header]
for class_name in class_names:
    row = [class_names.index(class_name), class_name]
    csvdat.append(row)

with open(output_dir + '/cars_meta.csv', 'w') as f:
    writer = csv.writer(f, dialect='myDialect')
    for row in csvdat:
        writer.writerow(row)
f.close()

print('Num of classes: ', len(class_names))
