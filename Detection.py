# from math import sqrt

import cv2
import numpy as np
import torch

import sys

sys.path.insert(0, './yolov5')

# from .yolov5.models.experimental import attempt_load
# from .yolov5.utils.downloads import attempt_download
# from .yolov5.models.common import DetectMultiBackend
# from .yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
# from .yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
#                                   check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
# from .yolov5.utils.torch_utils import select_device, time_sync
# from .yolov5.utils.plots import Annotator, colors, save_one_box
# from .deep_sort.utils.parser import get_config
# from .deep_sort.deep_sort import DeepSort


from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

from deep_sort.deep_sort import DeepSort

#
#
#
# from Yolov5_DeepSort_OSNet.yolov5.models.common import DetectMultiBackend
# A = DetectMultiBackend()
#
#

import constants


class ObjectsDetector:
    def __init__(self, config_file, weights_file, yolo=None):
        self.net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.yolov5 = yolo
        self.tracked_id = 0
        self.deep_sort_obj = DeepSort('osnet_x0_25', torch.device('cpu'), max_dist=0.2, max_iou_distance=0.7,
                                      max_age=30, n_init=3, nn_budget=100)

        # self.yolov5 = DetectMultiBackend('yolov5n.pt')

    """
    finds the bonding box with the highest probability with a person inside it [x, y, w, h]
    """

    def detect(self, img, prob=0.2):
        if not self.yolov5:
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            person = self.net.forward(self.ln)
            # p1,p2,p3 = person#todo
            p1, p2 = person
            p1 = p1[np.where(p1[:, 5] > prob)]
            p2 = p2[np.where(p2[:, 5] > prob)]
            # p3 = p3[np.where(p3[:, 5] > prob)]
            p1 = p1[:, :6]
            p2 = p2[:, :6]
            # p3 = p3[:,:6]
            if p1.shape[0] != 0:
                p1 = p1[np.argmax(p1[:, 5])]
                p1 = p1.reshape((-1, 6))
            if p2.shape[0] != 0:
                p2 = p2[np.argmax(p2[:, 5])]
                p2 = p2.reshape((-1, 6))
            # if p3.shape[0] != 0:
            #     p3 = p3[np.argmax(p3[:, 5])]
            #     p1 = p1.reshape((-1, 6))

            # person = np.vstack([p1,p2,p3])#todo
            person = np.vstack([p1, p2])
            if person.shape[0] != 0:
                person = person[np.argmax(person[:, 5])]
            person = person.reshape((-1, 6))
            person[:, 0:4] *= np.array([w, h, w, h])
            person[:, 0] -= person[:, 2] // 2
            person[:, 1] -= person[:, 3] // 2
            return person
        else:
            h, w = img.shape[:2]
            i = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            i = np.ascontiguousarray(i)

            i = i[np.newaxis, :, :, :] / 255
            i = torch.tensor(i).float()

            person = self.yolov5(i)
            person = non_max_suppression(person, conf_thres=prob, classes=[0],iou_thres=0.7)[0]
            xywhs = xyxy2xywh(person[:, 0:4])
            confs = person[:, 4]
            clss = person[:, 5]
            detections = self.deep_sort_obj.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img)
            if len(detections) > 0:
                detections[:, 0:4] = xyxy2xywh(detections[:, 0:4])
                return self.select_object_track(detections)
            return np.array([])

    def select_object_track(self, detections):
        """

        :param detections: [x, y, w, h, id, class, conf]
        :return: the object bbox = [x, y, w, h, id, class, conf] in order to track
        """
        if len(detections) > 0:
            ids = detections[:, 4]
            # if self.tracked_id in ids:
            #     return detections[np.where(self.tracked_id == ids)][0]
            # else:

            min_id = np.argmin(ids)
            self.tracked_id = ids[min_id]
            return detections[min_id]
        return None

    def find_center(self, img):
        """
        [x,y,w,h,cx,cy,area]
        """
        person = self.detect(img)
        if person.shape[0] != 0:
            # x = person[0][0]
            # y = person[0][1]
            # w = person[0][2]
            # h = person[0][3]
            # print(person)
            x = person[0]
            y = person[1]
            w = person[2]
            h = person[3]
            return [x, y, w, h, (x + w / 2), (y + h / 2),
                    (w * h) / constants.FRAME_SIZE]  # TODO maybe without returning x, y, w, h?
