# coding:utf-8

import sys

sys.path.append('..')

from face_cls.cls import rnet_cls
from face_cls.predict import Predict
from face_cls.RNet_models import *
from face_cls.dataLoader import TestLoader
import cv2
import os
import numpy as np

test_mode = "RNet"
thresh = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
slide_window = False
shuffle = False
nets = [None]
prefix = ['D:/Pycharm/project/RNet/data/model/model_0711.tar/model_0711/model_0711/RNet']
epoch = [30, 14, 16]
batch_size = [2048, 64, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]


# RNet = Predict(R_Net, 24, batch_size[1], model_path[0])
RNet = Predict(R_Net, 24, 1, model_path[0])
nets[0] = RNet
rnetCls = rnet_cls(cls_nets=nets, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
gt_imdb = []
# gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
# imdb_ = dict()"
# imdb_['image'] = im_path
# imdb_['label'] = 5
labels = []
image_dir = "D:/CompanyData/cropImage_cls/cropImage"
path = "D:/CompanyData/cropImage_cls/cropImage/label.txt"
outputRight = 'D:/CompanyData/cropImage_cls/cropImage/right'
outputWrong = 'D:/CompanyData/cropImage_cls/cropImage/wrong'
if not os.path.exists(outputRight):
    os.makedirs(outputRight)
if not os.path.exists(outputWrong):
    os.makedirs(outputWrong)
reader = open(path, 'r')
line = reader.readline().strip()
imgName = []
while line:
    item =line.split(' ')
    gt_imdb.append(os.path.join(image_dir, item[0]))
    imgName.append(item[0])
    labels.append(int(item[1]))
    line = reader.readline().strip()
test_data = TestLoader(gt_imdb)
probs = rnetCls.cls_face(test_data)
cls_prob = tf.convert_to_tensor(probs, tf.float32, name='cls_probs')
# print(cls_prob)
# cls_probs = np.squeeze(np.array(cls_probs))
# print(labels)
reader.close()

cls_probs = tf.squeeze(cls_prob)

num_row = tf.to_int32(cls_probs.get_shape()[0])
num_cls_prob = tf.size(cls_probs)
cls_prob_reshape = tf.reshape(cls_probs, [num_cls_prob, -1])
    #row = [0,2,4.....]
row = tf.range(num_row)*2
indices_ = row+1
label_probs = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
zeros = tf.zeros_like(labels)
ones = tf.ones_like(labels)
# label=-1 --> label=0net_factory
# pos -> 1, neg -> 0, others -> 0
label_last = tf.where(tf.greater(label_probs, 0.65), ones, zeros)

# acc = cal_accuracy(cls_probs, labels)
# loss = cls_ohem(cls_probs, labels)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_last, labels), tf.float32))

sess = tf.Session()
print(sess.run(label_last))
print(sess.run(accuracy_op))
# print('acc is %4f'% sess.run(acc))
# # print(sess.run(loss))
bigger = label_last.eval(session=sess)
# bigger = np.argmax(cls, axis=1)
# bigger = np.squeeze(bigger)
# print(bigger)

for i in range(0, len(labels)):
    path = gt_imdb[i]
    name1 = path.split('/')[-1]
    name2 = name1.split('.')[0]
    name3 = name2.split('_')[-1]
    img = cv2.imread(path)

    if bigger[i] != labels[i]:
        dst_dir = outputWrong
    else:
        dst_dir = outputRight
    dst_path = os.path.join(dst_dir, name3+'_src'+str(labels[i])+'_prd'+str(bigger[i])+'.jpg')
    print(dst_path)
    cv2.imwrite(dst_path, img)

