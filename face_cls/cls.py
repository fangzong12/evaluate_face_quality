import cv2
import time
import numpy as np
import sys

sys.path.append("../")


class rnet_cls(object):

    def __init__(self,
                 cls_nets,
                 min_face_size=20,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.79,
                 # scale_factor=0.709,#change
                 slide_window=False):

        self.rnet_net = cls_nets[0]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window

    def cls_rnet(self, im):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        cropped_ims = (cv2.resize(im, (24, 24)) - 127.5) / 128
        # cls_scores : num_data*2
        # reg: num_data*4
        # landmark: num_data*10
        cls_scores = self.rnet_net.predict(cropped_ims)
        return cls_scores;

    # use for video
    def cls(self, img):
        cls_prob = self.cls_rnet(img)
        return cls_prob

    def cls_face(self, test_data):
        cls_probs = []  # save each image's bboxes
        batch_idx = 0
        sum_time = 0
        t2_sum = 0
        num_of_img = test_data.size
        # test_data is iter_
        s_time = time.time()
        for databatch in test_data:
            # databatch(image returned)
            batch_idx += 1
            if batch_idx % 100 == 0:
                c_time = (time.time() - s_time )/100
                print("%d out of %d images done" % (batch_idx, test_data.size))
                print('%f seconds for each image' % c_time)
                s_time = time.time()
            im = databatch

            t = time.time()
            # ignore landmark
            cls_prob = self.cls_rnet(im)
            cls_probs.append(cls_prob)
            t2 = time.time() - t
            sum_time += t2
            t2_sum += t2
        print('num of images', num_of_img)
        print("time cost in average" +
            '{:.3f}'.format(sum_time/num_of_img) +
            'rnet {:.3f}'.format(t2_sum/num_of_img))
        return np.concatenate(cls_probs, axis=0)

    def cls_single_image(self, im):
        cls_prob = self.cls_rnet(im)
        return cls_prob
