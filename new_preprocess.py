import os
import shutil
import pandas
import cv2 as cv

input_dir = 'livedat/'
annos_dir = 'newannos/output/'
train_dst_folder = 'data/train/'
val_dst_folder = 'data/valid/'
test_dst_folder = 'data/test/'
img_width, img_height = 224, 224

seq_start = 1000


def get_sample_numbers(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        total += len(files)
    return total


def remove_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_image(fname, label, bbox_x1, bbox_y1, bbox_x2, bbox_y2, type, resize):
    global seq_start
    x1, y1, x2, y2 = bbox_x1, bbox_y1, bbox_x2, bbox_y2

    src_path = os.path.join(input_dir, fname)
    src_image = cv.imread(src_path)
    height, width = src_image.shape[:2]
    # margins of 16 pixels
    margin = 16
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(x2 + margin, width)
    y2 = min(y2 + margin, height)

    if type == 'train':
        dst_path = os.path.join(train_dst_folder, str(label))
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, str(seq_start)+'.jpg')
    elif type == 'valid':
        dst_path = os.path.join(val_dst_folder, str(label))
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, str(seq_start)+'.jpg')
    else:
        dst_path = os.path.join(test_dst_folder, str(label))
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, str(seq_start)+'.jpg')

    crop_image = src_image[y1:y2, x1:x2]
    if(resize):
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
    else:
        dst_img = crop_image
    cv.imwrite(dst_path, dst_img)
    seq_start = seq_start + 1


def process_data(csv_path, type, resize):
    column_titles = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
    data = pandas.read_csv(csv_path, encoding="utf-8")
    dict_of_list = {}
    for column in column_titles:
        dict_of_list[column] = data[column].tolist()

    for bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id, fname in zip(dict_of_list.get('bbox_x1'),
                                                                   dict_of_list.get('bbox_y1'),
                                                                   dict_of_list.get('bbox_x2'),
                                                                   dict_of_list.get('bbox_y2'),
                                                                   dict_of_list.get('class'),
                                                                   dict_of_list.get('fname')):

        save_image(fname, class_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, type, resize)


remove_folder('data/')
ensure_folder(train_dst_folder)
ensure_folder(val_dst_folder)
ensure_folder(test_dst_folder)

process_data(os.path.join(annos_dir, 'cars_train_annos.csv'), 'train', True)
process_data(os.path.join(annos_dir, 'cars_val_annos.csv'), 'valid', True)
process_data(os.path.join(annos_dir, 'cars_test_annos.csv'), 'test', False)

print('Train:', get_sample_numbers(train_dst_folder))
print('Validation:', get_sample_numbers(val_dst_folder))
print('Test:', get_sample_numbers(test_dst_folder))
