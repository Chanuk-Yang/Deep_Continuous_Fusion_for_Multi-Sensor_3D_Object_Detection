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


class Train(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_total = LossTotal()
        self.model = ObjectDetection_DCF().cuda()
        self.loss_value = None
        lr = 0.0001
        beta1 = 0.9
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))

    def one_step(self, lidar_voxel, camera_image, object_data):
        pred_cls, pred_reg = self.model(lidar_voxel, camera_image)
        self.loss_value = self.loss_total(object_data, pred_cls, pred_reg)
        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def get_loss_value(self, lidar_image, camera_image, object_data):
        pred_cls, pred_reg = self.model(lidar_image, camera_image)
        self.loss_value = self.loss_total(object_data, pred_cls, pred_reg)
        return self.loss_value.item(), pred_cls, pred_reg


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    num_epochs = 60
    dataset = CarlaDataset()
    print("training is ready")

    training = Train()
    data_length = len(dataset)
    for epoch in range(num_epochs):
        for i in range(len(dataset)):
            image_data = dataset[i]['image'].cuda()
            point_voxel = dataset[i]['pointcloud'].cuda()
            reference_bboxes = dataset[i]['bboxes'].cuda()
            training.one_step(point_voxel, image_data, reference_bboxes)
            if i % 100 == 0:
                print("training at ", i, "is processed")
            if i % 500 == 0:
                test_index = np.random.randint(len(dataset))
                image_data = dataset[test_index]['image'].cuda()
                point_voxel = dataset[test_index]['pointcloud'].cuda()
                reference_bboxes = dataset[test_index]['bboxes'].cuda()
                loss_value, pred_cls, pred_reg = training.get_loss_value(point_voxel, image_data, reference_bboxes)
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                      % (epoch, num_epochs, i, data_length, loss_value))
                save_image(pred_cls[0, 1, :, :], 'image/positive_image_{}_in_{}.png'.format(i, epoch))
                save_image(pred_cls[0, 0, :, :], 'image/negative_image_{}_in_{}.png'.format(i, epoch))
