#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:41:40 2019

@author: zsp
"""
import tensorflow as tf 
import os, cv2
import tqdm
import numpy as np

class dataset_util():
    def int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def int64_list_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    def bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def bytes_list_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    
    def float_list_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    cocox = 0, xmax = 0+2, ymin = 1, ymax = 1+3
    
    """
    bboxes = img_data['bbox']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[2]+bbox[0])
        ymin.append(bbox[1])
        ymax.append(bbox[3]+bbox[1])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_data['height']),
        'image/width': dataset_util.int64_feature(img_data['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(img_data['label']),
        'image/encoded': dataset_util.bytes_feature(img_data['pixel_data']),
#        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example


IMAGE_FEATURE_MAP = {
#    'image/width': tf.io.FixedLenFeature([], tf.int64),
#    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
#    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
}

def parse_tfrecord(tfrecord):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (416, 416))
    # get numpy from x_train
    label_idx = x['image/object/class/label']
    labels = tf.sparse.to_dense(label_idx)
    labels = tf.cast(labels, tf.float32)
    y_train = tf.stack([
        tf.sparse.to_dense(x['image/object/bbox/xmin']),
        tf.sparse.to_dense(x['image/object/bbox/ymin']),
        tf.sparse.to_dense(x['image/object/bbox/xmax'])+tf.sparse.to_dense(x['image/object/bbox/xmin']),
        tf.sparse.to_dense(x['image/object/bbox/ymax'])+tf.sparse.to_dense(x['image/object/bbox/ymin']), 
        labels], axis=1)
    paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)
    return x_train, y_train

def load_tfrecord_dataset(record_file):
    files = tf.data.Dataset.list_files(record_file)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x))
    
def dict_dataset_from_cocofile(file, image_dir):
    '''
    example
        img = open('/home/zsp/Pictures/驾驶证1.jpg','rb').read()
    img_data = {
      'pixel_data': img,
      'bboxes': [[0.2, .3, .4, .5], [.1, .3, .4, .5]],
      'labels': [23, 34]
    }
    '''
    import json
    with open(file) as f:
        f_load = json.load(f)
    annotation_list = f_load['annotations']
    images = f_load['images']
    img_all = {}
    for annotation in annotation_list:
#        img_data = {}
        image_id = annotation['image_id']
#        img_data['bbox'] = annotation['bbox']
#        img_data['label'] = annotation['category_id']
#        img_data['id'] = annotation['id']
        if image_id not in img_all:
            img_all[image_id] = {}
            img_all[image_id]['bbox'] = []
            img_all[image_id]['label'] = []            
        img_all[image_id]['bbox'].append(annotation['bbox'])
        img_all[image_id]['label'].append(annotation['category_id'])
    
    for image in images:
        image_id = image['id']
        if image_id in img_all:
            img_all[image_id]['filename'] = image['file_name']
            with open(os.path.join(image_dir,image['file_name']),'rb') as f:
                img_encoded = f.read()
            im = cv2.imread(os.path.join(image_dir,image['file_name']))
            w,h,_= im.shape
            try:
                img_all[image_id]['pixel_data'] = img_encoded
                img_all[image_id]['width'] = w
                img_all[image_id]['height'] = h
            except Exception as e:
                print(e)
                print(image['file_name'])
    return img_all

def main_data2TFRecord():
    file ='/home/zsp/newspace/data/coco/annotations/instances_val2017.json'
    image_dir = '/home/zsp/newspace/data/coco/val2017'
    img_data = dict_dataset_from_cocofile(file, image_dir)
    record_file = '/home/zsp/newspace/project/detector/yolov3-tf2/data/test.tfrecords'
    with tf.io.TFRecordWriter(record_file) as Writer:
        for data in tqdm.tqdm(img_data.keys()):
#            print(img_data[data])
#            print(img_data[data]['label'])
            if np.max(img_data[data]['label'])>80:
                continue
    #        print(data)
            try:
                tf_example = dict_to_coco_example(img_data[data])
                Writer.write(tf_example.SerializeToString())
            except Exception as e:
                print(e)
                print(img_data[data])
            

if __name__=='__main__':
    main_data2TFRecord()
#    record_file = '/home/zsp/newspace/project/detector/yolov3-tf2/data/test.tfrecords'
#    train_dataset = load_tfrecord_dataset(record_file)
#    train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
#    train_dataset = train_dataset.batch(64)
#    for img, label in train_dataset:
#        print(img.shape, label.shape)


