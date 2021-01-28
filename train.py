import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import numpy as np
import yaml

from data_import_carla import CarlaDataset
# from kitti import KittiDataset
from loss import LossTotal
from model import ObjectDetection_DCF
from test import Test
from KITTIDataset import KITTIDataset

class Train(nn.Module):
    def __init__(self, config):
        super(Train, self).__init__()
        device_id_source = config["cuda_visible_id"].split(",")
        device_id = [i for i in range(len(device_id_source))]
        self.loss_total = LossTotal(config)
        self.model = ObjectDetection_DCF(config).cuda()
        self.model = DDP(self.model, device_ids=device_id, output_device=0, find_unused_parameters=True)
        self.loss_value = None
        lr = config["learning_rate"]
        beta1 = config["beta1"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, 0.999))

    def one_step(self, lidar_voxel, camera_image, object_data, num_ref_box):
        pred = self.model(lidar_voxel, camera_image)
        pred_cls, pred_reg, pred_bbox_f = torch.split(pred,[4, 14, 14], dim=1)
        self.loss_value = self.loss_total(object_data, num_ref_box, pred_cls, pred_reg)
        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def get_loss_value(self, lidar_voxel, camera_image, object_data, num_ref_box):
        pred = self.model(lidar_voxel, camera_image)
        pred_cls, pred_reg, pred_bbox_f = torch.split(pred,[4, 14, 14], dim=1)
        self.loss_value = self.loss_total(object_data, num_ref_box, pred_cls, pred_reg)
        return self.loss_value.item(), pred_cls, pred_reg


if __name__ == '__main__':
    print(torch.cuda.is_available())
    CONFIG_PATH = "./config/"
    config_name = "config_carla.yaml"
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    
    device_id_source = config["cuda_visible_id"].split(",")
    device_id = [i for i in range(len(device_id_source))]
    os.environ['CUDA_VISIBLE_DEVICES'] = config["cuda_visible_id"]
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config["port_number"]
    torch.distributed.init_process_group(backend='nccl', world_size=1, rank=0)
    if config["dataset_name"] == "carla":
        dataset = CarlaDataset(config)
        dataset_test = CarlaDataset(config, mode="test", want_bev_image=True)
        print("carla dataset is used for training")
    elif config["dataset_name"] =="kitti":
        dataset = KITTIDataset()
        dataset_test = KITTIDataset(mode="test")
        print("kitti dataset is used for training")
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    train_sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=config["batch_size"],
                                          sampler=train_sampler)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=config["batch_size"],
                                          sampler=train_sampler_test)
    num_epochs = config["num_epoch"]
    training = Train(config)
    test = Test(training.model, config)
    data_length = len(data_loader)
    for epoch in range(num_epochs):
        torch.save(training.model.state_dict(), "./saved_model/" + config["saved_model_name"])
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
                print("="*50)
                print('[%d/%d][%d/%d]\tLoss: %.4f in traning dataset'
                      % (epoch, num_epochs, batch_ndx, data_length, loss_value))
                for batch_ndx_, sample_ in enumerate(data_loader_test):
                    image_data_ = sample_['image'].cuda()
                    point_voxel_ = sample_['pointcloud'].cuda()
                    reference_bboxes_ = sample_['bboxes'].cpu().clone().detach()
                    num_ref_bboxes_ = sample_["num_bboxes"]
                    bev_image_ = sample_["lidar_bev_2Dimage"]
                    test.get_eval_value_onestep(point_voxel_, image_data_, reference_bboxes_, num_ref_bboxes_)
                    test.save_feature_result(bev_image_, reference_bboxes_, num_ref_bboxes_, batch_ndx_, epoch)
                    if batch_ndx_ > 5:
                        print("accumulated number of true data is ", test.get_num_T())
                        print("accumulated number of positive data is ", test.get_num_P())
                        print("accumulated number of true positive data is ", test.get_num_TP_set())
                        break
                test.display_average_precision(plot_AP_graph=config["plot_AP_graph"])
                print("="*50)
                test.initialize_ap()
        for batch_ndx, sample in enumerate(data_loader_test):
            image_data = sample['image'].cuda()
            point_voxel = sample['pointcloud'].cuda()
            reference_bboxes = sample['bboxes'].cpu().clone().detach()
            num_ref_bboxes = sample["num_bboxes"]
            bev_image = sample["lidar_bev_2Dimage"]
            test.get_eval_value_onestep(point_voxel, image_data, reference_bboxes, num_ref_bboxes)
            test.save_feature_result(bev_image, reference_bboxes, num_ref_bboxes, batch_ndx, epoch)
            if batch_ndx > 10:
                print("accumulated number of true data is ", test.get_num_T())
                print("accumulated number of positive data is ", test.get_num_P())
                print("accumulated number of true positive data is ", test.get_num_TP_set())
                print("="*50)
                break
        test.display_average_precision(plot_AP_graph=config["plot_AP_graph"])
        print("="*50)
        test.initialize_ap()
        