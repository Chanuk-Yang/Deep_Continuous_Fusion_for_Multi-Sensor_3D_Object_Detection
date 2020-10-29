import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import h5py
import numpy as np


from loss import LossTotal
from model import LidarBackboneNetwork, ObjectDetection_DCF
from data_import import getIdDict, getOneStepData, getLidarImage, Voxelization, putBoundingBox


class Train(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_total = LossTotal()
        self.model = LidarBackboneNetwork().cuda()
        self.loss_value = None
        lr = 0.0001
        beta1 = 0.9
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))

    def one_step(self, lidar_voxel, object_data):
        pred_cls, pred_reg = self.model(lidar_voxel)

        self.loss_value = self.loss_total(object_data, pred_cls)
        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def get_loss_value(self, lidar_image, object_data):
        pred_cls, pred_reg = self.model(lidar_image)
        self.loss_value = self.loss_total(object_data, pred_cls)
        return self.loss_value.item(), pred_cls, pred_reg


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    num_epochs = 60
    dataroot = "../dataset/view-synth"

    print("reading hdf5 file...")
    file_list = os.listdir(dataroot)
    hdf5_files = {}
    for file in file_list:
        if file.split('.')[-1] == 'hdf5':
            if file.split('.')[-2] == 'bagtest':
                file_dir = os.path.join(dataroot, file)
                try:
                    hdf5_files[file] = h5py.File(file_dir, 'r')
                    print(file)
                except:
                    print(file + ' doesnt work. we except this folder')
    print("reading hdf5 end")



    print("training is ready")
    training = Train()
    for epoch in range(num_epochs):
        id_dict_train, id_dict_test = getIdDict(hdf5_files)
        data = hdf5_files[list(id_dict_train.keys())[0]]
        for file in id_dict_train:
            data_length = len(id_dict_train[file])
            for i in range(data_length):
                id = id_dict_train[file][i].strip()
                object_datas, lidar_data, image_data = getOneStepData(data, id)
                point_voxel = Voxelization(lidar_data).cuda()
                lidar_image = getLidarImage(lidar_data).cuda()  # parse lidar data to BEV lidar image
                training.one_step(point_voxel, object_datas)
                if i % 100 == 0:
                    print("training at ", i, "is processed")
                if i % 500 == 0:
                    test_index = np.random.randint(data_length)
                    id = id_dict_train[file][i].strip()
                    object_datas, lidar_data, image_data = getOneStepData(data, id)
                    lidar_image = getLidarImage(lidar_data).cuda()  # parse lidar data to BEV lidar image
                    point_voxel = Voxelization(lidar_data).cuda()
                    loss_value, pred_cls, pred_reg = training.get_loss_value(point_voxel, object_datas)
                    lidar_image_with_bboxes = putBoundingBox(lidar_image[0, 0], object_datas)
                    print('[%d/%d][%d/%d]\tLoss: %.4f'
                          % (epoch, num_epochs, i, data_length, loss_value))
                    save_image(pred_cls[0, 1, :, :], 'image/positive_image_{}_in_{}.png'.format(i, epoch))
                    save_image(pred_cls[0, 0, :, :], 'image/negative_image_{}_in_{}.png'.format(i, epoch))
                    save_image(lidar_image_with_bboxes, 'image/lidar_image_{}_in_{}.png'.format(i, epoch))
