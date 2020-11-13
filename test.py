from numpy import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import numpy as np

from data_import_carla import CarlaDataset
from loss import LossTotal
from model import LidarBackboneNetwork, ObjectDetection_DCF
from data_import import putBoundingBox


class Test:
    def __init__(self, pre_trained_net):
        self.net = pre_trained_net
        self.net.eval()
        self.loss_total = LossTotal()

    def get_eval_value(self, lidar_image, camera_image, object_data):
        pred_cls, pred_reg = self.net(lidar_image, camera_image)
        self.loss_value = self.loss_total(object_data, pred_cls, pred_reg)
        return self.loss_value.item(), pred_cls, pred_reg
    
    def NMS(self):
        a = 1
        
    def AP(self):
        b = 1
        
    def mAP(self):
        c = 1


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    dataset = CarlaDataset()
    print("dataset is ready")
    test_model = ObjectDetection_DCF().cuda()
    test = Test()
    data_length = len(dataset)
    test_index = np.random.randint(data_length)
    image_data = dataset[test_index]['image'].cuda()
    point_voxel = dataset[test_index]['pointcloud'].cuda()
    reference_bboxes = dataset[test_index]['bboxes'].cuda()
    loss_value, pred_cls, pred_reg = test.get_eval_value(point_voxel, image_data, reference_bboxes)
    print('Loss: %.4f'% (loss_value))
    
