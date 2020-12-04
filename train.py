import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import numpy as np
import argparse

from data_import_carla import CarlaDataset
# from kitti import KittiDataset
from loss import LossTotal
from model import LidarBackboneNetwork, ObjectDetection_DCF
from data_import import putBoundingBox
from test import Test

class Train(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_total = LossTotal()
        self.model = ObjectDetection_DCF().cuda()
        self.loss_value = None
        lr = 0.0001
        beta1 = 0.9
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))

    def one_step(self, lidar_voxel, camera_image, object_data, num_ref_box):
        pred_cls, pred_reg, pred_bbox_f = self.model(lidar_voxel, camera_image)
        self.loss_value = self.loss_total(object_data, num_ref_box, pred_cls, pred_reg)
        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def get_loss_value(self, lidar_image, camera_image, object_data, num_ref_box):
        pred_cls, pred_reg, pred_bbox_f = self.model(lidar_image, camera_image)
        self.loss_value = self.loss_total(object_data, num_ref_box, pred_cls, pred_reg)
        return self.loss_value.item(), pred_cls, pred_reg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deep continuous fusion training is doing')
    parser.add_argument('--data', type=str, default="carla", help='Data type, choose [carla] or [kitti]')
    parser.add_argument('--cuda', type=int, default=0, help="cuda visible device number. you can choose 0~7")
    args = parser.parse_args()
    dataset_category = args.data
    cuda_vis_dev_num = args.cuda

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_vis_dev_num)
    if dataset_category == "carla":
        dataset = CarlaDataset()
        dataset_test = CarlaDataset(mode="test")
        print("carla dataset is used for training")
    elif dataset_category =="kitti":
        dataset = KittiDataset()
        dataset_test = KittiDataset(mode="test")
        print("kitti dataset is used for training")
    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=2,
                                          shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=2,
                                          shuffle=True)
    num_epochs = 60
    training = Train()
    test = Test(training.model)
    data_length = len(dataset)
    for epoch in range(num_epochs):
        for batch_ndx, sample in enumerate(data_loader):
            image_data = sample['image'].cuda()
            point_voxel = sample['pointcloud'].cuda()
            reference_bboxes = sample["bboxes"].cuda()
            num_ref_bboxes = sample["num_bboxes"]
            training.one_step(point_voxel, image_data, reference_bboxes, num_ref_bboxes)
            if batch_ndx % 100 == 0:
                print("training at ", batch_ndx, "is processed")
            if batch_ndx % 500 == 0:
                test_index = np.random.randint(len(dataset))
                loss_value, _, _ = training.get_loss_value(point_voxel, image_data, reference_bboxes, num_ref_bboxes)
                print('[%d/%d][%d/%d]\tLoss: %.4f in traning dataset'
                      % (epoch, num_epochs, batch_ndx, data_length, loss_value))
        for batch_ndx, sample in enumerate(data_loader_test):
            image_data = sample['image'].cuda()
            point_voxel = sample['pointcloud'].cuda()
            reference_bboxes = sample['bboxes'].cuda()
            test.get_eval_value_onestep(point_voxel, image_data, reference_bboxes,num_ref_bboxes)
            if batch_ndx % 200 == 0:
                print("accumulated number of true data is ", test.get_num_T())
                print("accumulated number of positive data is ", test.get_num_P())
                print("="*50)
            if batch_ndx > 20:
                break
        test.display_average_precision(plot_AP_graph=True)
        test.initialize_ap()