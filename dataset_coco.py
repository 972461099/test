"""

dataset loader of coco
using tensorflow 2.0 recommended api

"""
from alfred.dl.tf.common import mute_tf
mute_tf()
#from pycocotools.coco import COCO
#from PIL import Image
#from random import shuffle
import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from loguru import logger as logging
#from alfred.utils.log import logger as logging
#from yolov3.models import yolo_anchors, yolo_anchor_masks
import matplotlib.pyplot as plt 

#this_dir = os.path.dirname(os.path.abspath(__file__))

IMAGE_FEATURE_MAP = {
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
#    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
}

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

def parse_tfrecord(tfrecord, normalize_box=False):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    img_h = tf.cast(x['image/width'], dtype=tf.float32)
    img_w = tf.cast(x['image/height'], dtype=tf.float32)
    x_train = tf.image.resize(x_train, (416, 416))
    label_idx = x['image/object/class/label']
    labels = tf.sparse.to_dense(label_idx)
    labels = tf.cast(labels, tf.float32)
    # NOTE: since we not normalize when gen tfrecord, normalize it now
    if normalize_box:
        y_train = tf.stack([
            tf.sparse.to_dense(x['image/object/bbox/xmin']) / img_w,
            tf.sparse.to_dense(x['image/object/bbox/ymin']) / img_h,
            tf.sparse.to_dense(x['image/object/bbox/xmax']) / img_w,
            tf.sparse.to_dense(x['image/object/bbox/ymax']) / img_h,
            labels,
        ], axis=1)
    else:
        y_train = tf.stack([
            tf.sparse.to_dense(x['image/object/bbox/xmin']),
            tf.sparse.to_dense(x['image/object/bbox/ymin']),
            tf.sparse.to_dense(x['image/object/bbox/xmax']),
            tf.sparse.to_dense(x['image/object/bbox/ymax']),
            labels,
        ], axis=1)
    paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)
    return x_train, y_train, img_w, img_h


def load_tfrecord_dataset(file_pattern, normalize_box=False):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, normalize_box=normalize_box))


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(open('./images/girl.png', 'rb').read(),
                                   channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [[0.18494931, 0.03049111, 0.9435849, 0.96302897, 0],
              [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
              [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
              ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)
    # h, w = x_train.shape[1], x_train.shape[2]
    # ori_img = x_train.numpy()[0]
    # for l in labels:
    #     if l[0] != 0:
    #         print(l)
    #         cv2.rectangle(ori_img, (int(l[0] * w), int(l[1] * h)),
    #                       (int(l[2] * w), int(l[3] * h)), (0, 255, 0), 2)
    # cv2.imshow('rr', ori_img)
    # cv2.waitKey(0)
    return tf.data.Dataset.from_tensor_slices((x_train, y_train))

# ------------------------- For transform inputs ----------------------------------------------
@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs, classes):
    """
    y_true here is y_train append a column with anchor_idx
    which indicates, every object(box) anchored to which anchor
    then you can using anchor to trace location and regression the precise offset to
    the anchor
    """
    
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]
    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 5+classes))
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0

    for i in range(N):
        for j in range(tf.shape(y_true)[1]):
            # iter every box of 100
            if tf.equal(y_true[i][j][2], 0):
                # pass width is 0
                continue
            # assign those anchor matched with grid size to their grid
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5],
                                                      tf.int32))
            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                # box centeroid
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                rep = np.zeros(5+classes, np.float32)
                c = y_true[i,j,4]
                indexes_re = tf.TensorArray(tf.int32, 1, dynamic_size=False)
                updates_re = tf.TensorArray(tf.float32, 1, dynamic_size=False)
                indexes_re = indexes_re.write(0,[[0],[1],[2],[3],[4],[5+c]])
#                updates_re = updates_re.write(0,[y_true[i][j][0:5],1, 1])
                updates_re = updates_re.write(0,[box[0], box[1], box[2], box[3], 1, 1])
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)
                reped = tf.tensor_scatter_nd_update(rep, indexes_re.stack(), updates_re.stack())
    
                indexes = indexes.write(
                                idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                        idx,tf.cast(reped,dtype=tf.float32))
                idx += 1
                
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(),
                                       updates.stack())
#import librosa

'''
    
 [[0.203774512 0.320555538 0.81300652 0.903578401 18 8]
  [0.0970588252 0.0829575136 0.916470587 0.721781075 4 8]
  [0.471944422 0.114411756 0.516911745 0.158104584 47 1]

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
				 # 设定数据
                    print(l,b,i,j,k,c)
				 # 将T个box的标的数据统一放置到3*B*W*H*3的维度上。
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1
'''

def transform_targets(y_train, anchors, anchor_masks, classes):
    y_outs = []
    grid_size = 13
    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)
    y_train = tf.concat([y_train, anchor_idx], axis=-1)
#    print('y_train',tf.print(y_train))
    for anchor_idxs in anchor_masks:
#        print(anchor_idxs)
        one_grid_gt = transform_targets_for_output(y_train, grid_size, anchor_idxs,
                                         classes)
        
        y_outs.append(one_grid_gt)
        grid_size *= 2

    return tuple(y_outs)
    # return (box_wh, one_grid_gt)




def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255.
    return x_train


def transform_images_letterbox(x_train, size):
    """
    using letterbox to resize image, not strech image itself.
    resize by min(416, w, h), and padding rest pixel as 128
    """
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255.
    return x_train


if __name__ == "__main__":
    path = '/home/zsp/newspace/project/detector/yolov3-tf2/data/test.tfrecords'
    train_ds = load_tfrecord_dataset(path, True)
    train_dataset = train_ds.batch(1)
#    for x,y,w,h in train_dataset:
#        print(x,y,w,h)
#        tf.print(tf.reduce_max(y[:,:,4],axis=-1))
    train_dataset = train_dataset.map(lambda x, y, w, h: (transform_images(
        x, 416), transform_targets(y, yolo_anchors, yolo_anchor_masks, 80)))
    vis_data = False
    logging.info('dataset reading complete.')

    if vis_data:
        for img, label, w, h in train_ds.take(3):
            w, h = w.numpy(), h.numpy()
            img = np.array(img.numpy(), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_copy = img.copy()
            labels = np.array([i for i in label.numpy() if i[0] != 0])
            labels *= np.array([[w, h, w, h, 1]])
            labels = np.array(labels, dtype=np.int64)
            print(labels)

            for l in labels:
                cv2.rectangle(img_copy, (l[0], l[1]), (l[2], l[3]),
                              (255, 0, 255), 2)
            img_copy_refine = img.copy()
            # draw refined box on resized image
            scale_x = 416 / w
            scale_y = 416 / h
            for l in labels:
                cv2.rectangle(img_copy_refine,
                              (int(l[0] * scale_x), int(l[1] * scale_y)),
                              (int(l[2] * scale_x), int(l[3] * scale_y)),
                              (255, 0, 255), 2)
            ori_img = cv2.resize(img, (w, h))
            for l in labels:
                cv2.rectangle(ori_img, (l[0], l[1]), (l[2], l[3]), (0, 255, 0),
                              2)
            cv2.imshow('wrong', img_copy)
            cv2.imshow('ori_img', ori_img)
            cv2.imshow('img_copy_refine', img_copy_refine)
            cv2.waitKey(0)
            # break
    else:
        for img, label in train_dataset.take(10):
#            tf.print('0',tf.reduce_sum(label[0]))
#            tf.print('1',tf.reduce_sum(label[1]))
#            tf.print('2',tf.reduce_sum(label[2]))
#            print(label[0].shape)
            print(1)

#for i in range(13):
#    for j in range(26):
#        print(i,j,sum(sum(y[1][0][i][j])))
'''
[0, 8, 12, 2]
[2, 1, 6, 1]
[2, 1, 6, 1]
[0, 16, 24, 2]
[2, 3, 12, 1]
[2, 3, 13, 1]
[0, 33, 48, 2]
[2, 7, 25, 1]
[2, 7, 27, 1]

label[0][2, 1, 6, 1]
'''
