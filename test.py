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
        plot_AP_graph (False)
        """
        
        self.net = pre_trained_net
        self.net.eval()
        # self.loss_total = LossTotal()
        self.num_TP_set = {}
        self.num_TP_set_per_predbox = []
        self.num_T = 0
        self.num_P = 0
        self.IOU_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        for iou_threshold in self.IOU_threshold:
            self.num_TP_set[iou_threshold] = 0

    def get_num_T(self):
        return self.num_T

    def get_num_P(self):
        return self.num_P

    def get_eval_value_onestep(self, lidar_image, camera_image, object_data, plot_bev_image=False):

        start = time.time()
        pred_cls, pred_reg, pred_bbox_f = self.net(lidar_image, camera_image)
        inter_1 = time.time()
        # print("inference algorithm time :", inter_1 - start, "s")   
        pred_cls = pred_cls.cpu().clone().detach()
        pred_bbox_f = pred_bbox_f.cpu().clone().detach()

        inter_2 = time.time()
        # print("device to host time :", inter_2 - inter_1, "s")   ## its toooooooo slow
  
        pred_bboxes = self.get_bboxes(pred_cls, pred_bbox_f, score_threshold=0.7)
        inter_3 = time.time()
        # print("get_bboxes time :", inter_3 - inter_2, "s")    
        refined_bbox = self.NMS_IOU(pred_bboxes, nms_iou_score_theshold=0.01)
        # refined_bbox = self.NMS_SAT(pred_bboxes)
        # print("NMS time :", time.time() - inter_3, "s") 
        # print("total algorithm time :", time.time() - start, "s")
        # print("=" * 50)
        # self.loss_value = self.loss_total(object_data, pred_cls, pred_reg)
        # print(refined_bbox)

        if plot_bev_image:
            lidar_image_with_bboxes = putBoundingBox(lidar_image, refined_bbox[0])
            save_image(lidar_image_with_bboxes, 'image/lidar_image.png')
        
        self.precision_recall_singleshot(refined_bbox, object_data) # single batch
        # return self.loss_value.item(), pred_cls, pred_reg
    
    def get_bboxes_sort(self, pred_cls, pred_reg, selet_bbox_threshold=50):        
        
        B, C, W, H = pred_cls.shape
        selected_bboxes_batch =[]
        for b in range(B):
            # need to change two anchor at pred_cls and pred_reg
            start = time.time()
            sorted, indices = torch.sort(pred_cls[b,1].reshape(-1), descending=True)
            # print("sort time :", time.time() - start, "s")   
            row = indices // H
            col = indices % H
            selected_bboxes = pred_reg[b, :7, row[:selet_bbox_threshold], col[:selet_bbox_threshold]]
            selected_bboxes_batch.append(selected_bboxes.permute((1,0)))
        return selected_bboxes_batch
    
    
    
    def get_bboxes(self, pred_cls, pred_reg, score_threshold=0.7):
        """
        get bounding box score threshold instead of selecting bounding box
        """
        B, C, W, H = pred_cls.shape
        selected_bboxes_batch =[]
        for b in range(B):
            # need to change two anchor at pred_cls and pred_reg
            start = time.time()
            pred_cls_= pred_cls[b,1].view(-1) > score_threshold


            indices = torch.nonzero(pred_cls_).view(-1)
            # print("bbox time: ", time.time() - start)

            pred_reg_ = pred_reg[b, :7, :, :].view((7,-1))
            selected_bboxes = pred_reg_[:,indices].permute(1,0)
            selected_bboxes_batch.append(selected_bboxes)
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
                center = bbox[:3].numpy()
                box_size = bbox[3:6].numpy()
                heading_angle = bbox[6].numpy()
                cand_bbox_corners = get_3d_box(center, box_size, heading_angle)
                for selected_bbox in filtered_bboxes:
                    center_ = selected_bbox[:3].numpy()
                    box_size_ = selected_bbox[3:6].numpy()
                    heading_angle_ = selected_bbox[6].numpy()
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
            if pred_bboxes[b].shape[0] == 0:
                filtered_bboxes_batch.append(None)
                continue
            for i in range(pred_bboxes[b].shape[0]):
                bbox = pred_bboxes[b][i]
                if len(filtered_bboxes) == 0:
                    filtered_bboxes.append(bbox)
                    continue
                center = bbox[:3].numpy()
                box_size = bbox[3:6].numpy()
                heading_angle = bbox[6].numpy()
                cand_bbox_corners = get_vertice_rect(center, box_size, heading_angle)
                for selected_bbox in filtered_bboxes:
                    center_ = selected_bbox[:3].numpy()
                    box_size_ = selected_bbox[3:6].numpy()
                    heading_angle_ = selected_bbox[6].numpy()
                    selected_bbox_corners = get_vertice_rect(center_, box_size_, heading_angle_)
                    is_overlapped = separating_axis_theorem(cand_bbox_corners, selected_bbox_corners)
                    if is_overlapped:
                        filtered_bboxes_index.append(i)
            for ind in filtered_bboxes_index:
                filtered_bboxes.append(pred_bboxes[b][ind])
            filtered_bboxes_batch.append(filtered_bboxes)
        return filtered_bboxes_batch
        
        
    def precision_recall_singleshot(self, pred_bboxes, ref_bboxes):
        B,_,_ = ref_bboxes.shape
        for b in range(B):
            pred_bboxes_sb = pred_bboxes[b]
            ref_bboxes_sb = ref_bboxes[b]
            if pred_bboxes_sb != None:
                for pred_bbox in pred_bboxes_sb:
                    self.num_P += 1
                    center = pred_bbox[:3].numpy()
                    box_size = pred_bbox[3:6].numpy()
                    heading_angle = pred_bbox[6].numpy()
                    pred_bbox_corners = get_3d_box(center, box_size, heading_angle)
                    true_positive_cand_score = {}
                    for ref_bbox in ref_bboxes_sb:
                        if ref_bbox[-1] == 1:
                            center_ = ref_bbox[:3].numpy()
                            box_size_ = ref_bbox[3:6].numpy()
                            heading_angle_ = ref_bbox[6].numpy()
                            ref_bbox_corners = get_3d_box(center_, box_size_, heading_angle_)
                            (IOU_3d, IOU_2d) = box3d_iou(pred_bbox_corners, ref_bbox_corners)
                            for iou_threshold in self.IOU_threshold:
                                if IOU_3d > iou_threshold:
                                    true_positive_cand_score[iou_threshold] = IOU_3d
                    
                    for iou_threshold in self.IOU_threshold:
                        if iou_threshold in true_positive_cand_score:
                            self.num_TP_set[iou_threshold] += 1
                    self.num_TP_set_per_predbox.append(self.num_TP_set)
            for ref_bbox_ in ref_bboxes_sb:
                if ref_bbox_[-1] == 1:
                    self.num_T += 1

        
    def display_average_precision(self, plot_AP_graph=False):
        """
        need to IOU threshold varying 
        """
        total_precision = {}
        total_recall = {}
        for iou_threshold in self.IOU_threshold:
            total_precision[iou_threshold] = self.num_TP_set[iou_threshold] / (self.num_P + 0.00001)
            total_recall[iou_threshold] = self.num_TP_set[iou_threshold] / (self.num_T + 0.00001)

        print("Total Precision: ", total_precision)
        print("Total Recall: ", total_recall)

        precisions = {}
        recalls = {}
        num_P = 0
        for iou_threshold in self.IOU_threshold:
            precisions[iou_threshold] = [1]
            recalls[iou_threshold] = [0]
        for num_tp_set in self.num_TP_set_per_predbox:
            num_P+=1
            for iou_threshold in self.IOU_threshold:
                precisions[iou_threshold].append(num_tp_set[iou_threshold] / num_P)
                recalls[iou_threshold].append(num_tp_set[iou_threshold] / self.num_T)
        print("precisions: ", precisions)
        print("recalls: ", recalls)
        if plot_AP_graph:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            lines = []
            for iou_threshold in self.IOU_threshold:
                line = 0
                if len(recalls[iou_threshold]) > 1: 
                    line = ax.plot(recalls[iou_threshold], precisions[iou_threshold])
                else:
                    line = ax.plot([0,0])
                lines.append(line)

            fig.legend(lines, labels=self.IOU_threshold, title="IOU threshold value")
            fig.savefig('ap_result/test.png')

        



    def initialize_ap(self):
        self.num_TP_set = {}
        self.num_T = 0
        self.num_P = 0
        self.num_TP_set_per_predbox = []
        for iou_threshold in self.IOU_threshold:
            self.num_TP_set[iou_threshold] = 0

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Focus on test dataset
    dataset = CarlaDataset(mode="test")
    print("dataset is ready")
    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=4,
                                          shuffle=True)
    # Load pre-trained model. you can use the model during training instead of test_model 
    test_model = ObjectDetection_DCF().cuda()
    test = Test(test_model)
    data_length = len(dataset)
    loss_value = None

    for batch_ndx, sample in enumerate(data_loader):
        print("batch_ndx is ", batch_ndx)
        print("sample keys are ", sample.keys())
        print("bbox shape is ", sample["bboxes"].shape)
        print("image shape is ", sample["image"].shape)
        print("pointcloud shape is ", sample["pointcloud"].shape)
        test_index = np.random.randint(data_length)
        image_data = sample['image'].cuda()
        point_voxel = sample['pointcloud'].cuda()
        reference_bboxes = sample['bboxes'].cuda()
        
        # evaluate AP in one image and voxel lidar
        test.get_eval_value_onestep(point_voxel, image_data, reference_bboxes)
        print("accumulated number of true data is ", test.get_num_T())
        print("accumulated number of positive data is ", test.get_num_P())
        print("="*50)
        if batch_ndx > 10:
            break

    # display average-precision plot and mAP
    test.display_average_precision(plot_AP_graph=True)
    # MUST DO WHEN U DISPLAY ALL OF RESULTS
    test.initialize_ap()
    
