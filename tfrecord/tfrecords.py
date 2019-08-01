# coding:utf-8

import os
import random
import sys
import time
import tensorflow as tf
from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.
    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    # print('---', filename)
    # imaga_data:array to string
    # height:original image's height
    # width:original image's width
    # image_example dict contains image's info
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, net):
    # st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    # return '%s/train_PNet_landmark.tfrecord' % (output_dir)
    return '%s/%s.tfrecord' % (output_dir, name)


def run(dataset_dir, net, output_dir, name='RNet', shuffling=False):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    # tfrecord name
    tf_filename = _get_output_filename(output_dir, name, net)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    dataset = get_dataset(dataset_dir, net=net)
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        # andom.seed(12345454)
        random.shuffle(dataset)
        random.shuffle(dataset)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    print('lala')
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i + 1) % 100 == 0:
                sys.stdout.write('\r>> %d/%d images has been converted,label_hat is %d, label_mask is %d, '
                                 'label_block is %d, label_blur is %d, label_bow is %d, label_illumination is %d' %
                                 (i + 1, len(dataset), image_example['label_hat'], image_example['label_mask'],
                image_example['label_block'], image_example['label_blur'], image_example['label_bow'],
                                  image_example['label_illumination']))
            sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(dataset_dir+'/'+filename, image_example, tfrecord_writer)
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the RNet dataset!')


def get_dataset(dir, net="RNet"):
    '''
    :param dir: directory of the raw data
    :param net:
    :return:
    '''
    # item = 'imglists/PNet/train_%s_raw.txt' % net
    # item = 'imglists/PNet/train_%s_landmark.txt' % net
    item = 'CelebA/list/train_RNet_full.txt'
    dataset_dir = os.path.join(dir, item)
    print('dataset dir is :', dataset_dir)
    imagelist = open(dataset_dir, 'r')
    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        data_example['filename'] = info[0]
        data_example['label_hat'] = int(info[1])
        data_example['label_mask'] = int(info[2])
        data_example['label_block'] = int(info[3])
        data_example['label_blur'] = int(info[4])
        data_example['label_bow'] = int(info[5])
        data_example['label_illumination'] = int(info[6])
        dataset.append(data_example)
    return dataset


if __name__ == '__main__':
    # dir = './data/image0716'
    dir = 'D:/Pycharm/project/FaceAttribute-FAN-master/FaceAttribute-FAN-master/data'
    net = 'RNet0801'
    # output_directory = './data/tfrecord0716'
    output_directory = 'D:/CompanyData'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    name = 'RNet'
    run(dir, net, output_directory, name, shuffling=True)
