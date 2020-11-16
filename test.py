from numpy import random
from numpy.core.fromnumeric import argmax
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
from IOU import get_3d_box, box3d_iou
from separation_axis_theorem import get_vertice_rect, separating_axis_theorem

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


class Test:
    def __init__(self, pre_trained_net):
        """
        configuration
        
        plot_bev_image (False)
        selet_bbox_threshold (50)
        nms_iou_score_theshold (0.01)
        ap_iou_score_threshold (0.5)
        plot_AP_graph (False)
        """
        
        self.net = pre_trained_net
        self.net.eval()
        self.loss_total = LossTotal()
        self.num_TP = 0
        self.num_T = 0
        self.num_P = 0
        self.precisions = []
        self.recalls = []

    def get_eval_value_onestep(self, lidar_image, camera_image, object_data, plot_bev_image=False):

        start = time.time()
        pred_cls, pred_reg = self.net(lidar_image, camera_image)
        inter_1 = time.time()
        print("inference algorithm time :", inter_1 - start, "s")     
        pred_bboxes = self.get_bboxes(pred_cls, pred_reg, selet_bbox_threshold=50)
        inter_2 = time.time()
        # print("get_bboxes time :", inter_2 - inter_1, "s")    
        # refined_bbox = self.NMS_IOU(pred_bboxes, nms_iou_score_theshold=0.01)
        refined_bbox = self.NMS_SAT(pred_bboxes)
        print("NMS time :", time.time() - inter_2, "s") 
        print("total algorithm time :", time.time() - start, "s")
        print("=" * 50)
        self.loss_value = self.loss_total(object_data, pred_cls, pred_reg)
        

        if plot_bev_image:
            lidar_image_with_bboxes = putBoundingBox(lidar_image, refined_bbox[0])
            save_image(lidar_image_with_bboxes, 'image/lidar_image.png')
        self.precision_recall_singleshot(refined_bbox[0], object_data, ap_iou_score_threshold=0.5) # single batch
        return self.loss_value.item(), pred_cls, pred_reg
    
    def get_bboxes(self, pred_cls, pred_reg, selet_bbox_threshold=50):
        B, C, W, H = pred_cls.shape
        selected_bboxes_batch =[]
        for b in range(B):
            # need to change two anchor at pred_cls and pred_reg
            start = time.time()
            sorted, indices = torch.sort(pred_cls[b,1].reshape(-1), descending=True)
            print("sort time :", time.time() - start, "s")   
            row = indices // H
            col = indices % H
            selected_bboxes = pred_reg[b, :7, row[:selet_bbox_threshold], col[:selet_bbox_threshold]]
            selected_bboxes_batch.append(selected_bboxes.permute((1,0)))
        return selected_bboxes_batch
        
        

    def NMS_IOU(self, pred_bboxes, nms_iou_score_theshold=0.01):
        filtered_bboxes_batch = []
        B = len(pred_bboxes)
        for b in range(B):
            filtered_bboxes = []
            filtered_bboxes_index = []
            for i in range(pred_bboxes[b].shape[0]):
                bbox = pred_bboxes[b][i]
                if len(filtered_bboxes) == 0:
                    filtered_bboxes.append(bbox)
                    continue
                center = bbox[:3].cpu().clone().detach().numpy()
                box_size = bbox[3:6].cpu().clone().detach().numpy()
                heading_angle = bbox[6].cpu().clone().detach().numpy()
                cand_bbox_corners = get_3d_box(center, box_size, heading_angle)
                for selected_bbox in filtered_bboxes:
                    center_ = selected_bbox[:3].cpu().clone().detach().numpy()
                    box_size_ = selected_bbox[3:6].cpu().clone().detach().numpy()
                    heading_angle_ = selected_bbox[6].cpu().clone().detach().numpy()
                    selected_bbox_corners = get_3d_box(center_, box_size_, heading_angle_)
                    (IOU_3d, IOU_2d) = box3d_iou(cand_bbox_corners, selected_bbox_corners)
                    if IOU_3d < nms_iou_score_theshold:
                        filtered_bboxes_index.append(i)
            for ind in filtered_bboxes_index:
                filtered_bboxes.append(pred_bboxes[b][ind])
            filtered_bboxes_batch.append(filtered_bboxes)
        return filtered_bboxes_batch
        
    def NMS_SAT(self, pred_bboxes):
        # IOU vs SAT(separate axis theorem)
        filtered_bboxes_batch = []
        B = len(pred_bboxes)
        for b in range(B):
            filtered_bboxes = []
            filtered_bboxes_index = []
            for i in range(pred_bboxes[b].shape[0]):
                bbox = pred_bboxes[b][i]
                if len(filtered_bboxes) == 0:
                    filtered_bboxes.append(bbox)
                    continue
                center = bbox[:3].cpu().clone().detach().numpy()
                box_size = bbox[3:6].cpu().clone().detach().numpy()
                heading_angle = bbox[6].cpu().clone().detach().numpy()
                cand_bbox_corners = get_vertice_rect(center, box_size, heading_angle)
                for selected_bbox in filtered_bboxes:
                    center_ = selected_bbox[:3].cpu().clone().detach().numpy()
                    box_size_ = selected_bbox[3:6].cpu().clone().detach().numpy()
                    heading_angle_ = selected_bbox[6].cpu().clone().detach().numpy()
                    selected_bbox_corners = get_vertice_rect(center_, box_size_, heading_angle_)
                    is_overlapped = separating_axis_theorem(cand_bbox_corners, selected_bbox_corners)
                    if is_overlapped:
                        filtered_bboxes_index.append(i)
            for ind in filtered_bboxes_index:
                filtered_bboxes.append(pred_bboxes[b][ind])
            filtered_bboxes_batch.append(filtered_bboxes)
        return filtered_bboxes_batch
        
        
    def precision_recall_singleshot(self, pred_bboxes, ref_bboxes, ap_iou_score_threshold=0.5):
        for pred_bbox in pred_bboxes:
            self.num_P += 1
            center = pred_bbox[:3].cpu().clone().detach().numpy()
            box_size = pred_bbox[3:6].cpu().clone().detach().numpy()
            heading_angle = pred_bbox[6].cpu().clone().detach().numpy()
            pred_bbox_corners = get_3d_box(center, box_size, heading_angle)
            true_positive_cand = []
            true_positive_cand_score = []
            for ref_bbox in ref_bboxes:
                center_ = ref_bbox[:3].cpu().clone().detach().numpy()
                box_size_ = ref_bbox[3:6].cpu().clone().detach().numpy()
                heading_angle_ = ref_bbox[6].cpu().clone().detach().numpy()
                ref_bbox_corners = get_3d_box(center_, box_size_, heading_angle_)
                (IOU_3d, IOU_2d) = box3d_iou(pred_bbox_corners, ref_bbox_corners)
                if IOU_3d > ap_iou_score_threshold:
                    true_positive_cand.append(pred_bbox)
                    true_positive_cand_score.append(IOU_3d)
                if len(true_positive_cand_score) > 0:
                    self.num_TP += 1
            self.precisions.append(self.num_TP/self.num_P)
            self.recalls.append(self.num_TP)
        for ref_bbox_ in ref_bboxes:
            self.num_T += 1

        
    def display_average_precision(self, plot_AP_graph=False):
        total_precision = self.num_TP / self.num_P
        total_recall = self.num_TP / self.num_T

        self.recalls = [x / self.num_T for x in self.recalls ]

        if plot_AP_graph:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot([1,2,3])
            fig.savefig('ap_result/test.png')

        print("Precision: ", total_precision)
        print("Recall: ", total_recall)


    def initialize_ap(self):
        self.num_TP = 0
        self.num_T = 0
        self.num_P = 0
        self.precisions = []
        self.recalls = []



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    dataset = CarlaDataset()
    print("dataset is ready")
    test_model = ObjectDetection_DCF().cuda()
    test = Test(test_model)
    data_length = len(dataset)
    loss_value = None
    for i in range(10):
        test_index = np.random.randint(data_length)
        image_data = dataset[test_index]['image'].cuda()
        point_voxel = dataset[test_index]['pointcloud'].cuda()
        reference_bboxes = dataset[test_index]['bboxes'].cuda()
        loss_value, pred_cls, pred_reg = test.get_eval_value_onestep(point_voxel, image_data, reference_bboxes)
    test.display_average_precision()
    test.initialize_ap()
    print('Loss: %.4f'% (loss_value))
    
