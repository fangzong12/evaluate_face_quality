#coding:utf-8
from rnet.RNet_models import R_Net
from rnet.train import train
import os
import sys


def train_RNet(base_dir, prefix, end_epoch, logDir, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = R_Net
    train(net_factory, prefix, end_epoch, base_dir, logDir, display=display, base_lr=lr)


if __name__ == '__main__':
    base_dir = '/nfs/data/DRG/fz'
    model_name = 'RNet'
    model_path = '/nfs/data/DRG/fz/model_0711/%s' % model_name
    prefix = model_path
    logDir = str(sys.argv[-2])
    print(logDir)

    end_epoch = 44
    display = 100
    lr = 0.001
    train_RNet(base_dir, prefix, end_epoch, logDir, display, lr)
