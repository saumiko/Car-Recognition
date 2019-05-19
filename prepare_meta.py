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


# Make sure all classes exist in training set
def check_all_class_in_training_set():
    train_annos = annotations[:training]
    train_classes = []
    for annotation in train_annos:
        if annotation['class'] not in train_classes:
            train_classes.append(annotation['class'])
    if(len(train_classes) == len(class_names)):
        return False
    else:
        return True


while(check_all_class_in_training_set()):
    shuffle(annotations)


# Write full processed meta data in 60, 20, 20 ratio
csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)


def percentage(percent, whole):
  return int((percent * whole) / 100)


# Training
header = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
csvdat = [header]
ckpt = 0
for i in range(ckpt, percentage(training, len(annotations))):
    row = annotations[i]
    rowdat = [
        row['bbox_x1'],
        row['bbox_y1'],
        row['bbox_x2'],
        row['bbox_y2'],
        row['class'],
        row['fname']
    ]
    csvdat.append(rowdat)

with open(output_dir+'/cars_train_annos.csv', 'w') as f:
    writer = csv.writer(f, dialect='myDialect')
    for row in csvdat:
        writer.writerow(row)
f.close()


# Validation
header = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
csvdat = [header]
ckpt = percentage(training, len(annotations))
for i in range(ckpt, ckpt+percentage(validation, len(annotations))):
    row = annotations[i]
    rowdat = [
        row['bbox_x1'],
        row['bbox_y1'],
        row['bbox_x2'],
        row['bbox_y2'],
        row['fname']
    ]
    csvdat.append(rowdat)

with open(output_dir + '/cars_val_annos.csv', 'w') as f:
    writer = csv.writer(f, dialect='myDialect')
    for row in csvdat:
        writer.writerow(row)
f.close()


# Test
header = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
csvdat = [header]
ckpt = ckpt+percentage(validation, len(annotations))
for i in range(ckpt, ckpt+percentage(test, len(annotations))):
    row = annotations[i]
    rowdat = [
        row['bbox_x1'],
        row['bbox_y1'],
        row['bbox_x2'],
        row['bbox_y2'],
        row['class'],
        row['fname']
    ]
    csvdat.append(rowdat)

with open(output_dir + '/cars_test_annos.csv', 'w') as f:
    writer = csv.writer(f, dialect='myDialect')
    for row in csvdat:
        writer.writerow(row)
f.close()


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
