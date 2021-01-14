import torch
import torch.nn as nn
from torchvision.utils import save_image

import numpy as np
import random
import torch.nn.functional as F
import os

from IOU import get_3d_box, box3d_iou
from model import AnchorBoundingBoxFeature


class LossClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, predicted_feature, binary_label):
        loss = self.loss(predicted_feature, binary_label)
        return loss

class LossReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.SmoothL1Loss(reduction="none")

    def forward(self, predicted_feature, binary_label):
        loss = self.loss(predicted_feature, binary_label)
        return loss


class LossTotal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_class = LossClass()
        self.loss_regress = LossReg()
        self.anchor_bbox_feature = AnchorBoundingBoxFeature(config)
        anchor_set = self.anchor_bbox_feature().cuda()
        anchor_set_shape = anchor_set.shape
        self.anchor_set = anchor_set.reshape(2,int(anchor_set_shape[0]/2),
                                            anchor_set_shape[1],anchor_set_shape[2])
        self.regress_type = self.config["regress_type"]

    def forward(self, reference_bboxes_batch, num_ref_bbox_batch, 
                predicted_class_feature_batch, predicted_regress_feature_batch):
        # reference_bboxes : B, max_n(20), 8
        B, max_num, _ = reference_bboxes_batch.shape
        total_loss = torch.zeros(1).cuda()
        self.anchor_set_ = self.anchor_set.unsqueeze(0).repeat(B,1,1,1,1)
        for b in range(B):
            reference_bboxes = reference_bboxes_batch[b,:num_ref_bbox_batch[b]]
            predicted_class_feature = predicted_class_feature_batch[b]
            predicted_regress_feature = predicted_regress_feature_batch[b]
            anchor = self.anchor_set_[b]

            IDX, positive_position_list_all, positive_position_list = self.getPositionOfPositive(anchor,reference_bboxes,
                                                                                                sample_threshold=self.config["pos_sample_threshold"])
            negative_position_list = self.getPositionOfNegative(anchor, positive_position_list,
                                                                sample_threshold=self.config["neg_sample_threshold"])
            # positive_position_list = [[1,2], [3,4], [11,12], [13,14]]
            # negative_position_list = [[5,6], [7,8], [9,10]]
            total_loss_class = self.getClassSum(positive_position_list, negative_position_list, predicted_class_feature[:2,:,:], self.loss_class)    # per anchor
            total_loss_class += self.getClassSum(positive_position_list, negative_position_list, predicted_class_feature[2:4,:,:], self.loss_class)  # per anchor

            ## anchor의 좌표를 절대 좌표로 바꿔야 한다.
            ## position_list_all은 positive_position이 다 들어간것, true 하나당 한 픽셀이려면 수정 필요

            Reg_loss = self.getRegSum(IDX, positive_position_list_all, reference_bboxes, predicted_regress_feature, anchor)
            total_loss = total_loss_class + self.config["regress_loss_gain"] * Reg_loss
        return total_loss

    def getPositionOfPositive(self, anchor_bbox_feature, ref_bboxes, sample_threshold = 128):
        _, C, H, W = anchor_bbox_feature.shape
        positive_position_list = []
        positive_position_regress = []
        positive_position_idx = {}
        temp_cnt = 0
        for i, ref_bbox in enumerate(ref_bboxes):
            positive_position_idx[i] = []
            x_scale = int(self.config["voxel_length"] / (self.config["lidar_x_max"] - self.config["lidar_x_min"]))
            y_scale = int(self.config["voxel_width"] / (self.config["lidar_y_max"] - self.config["lidar_y_min"]))
            x_offset = int(-self.config["lidar_x_min"] * x_scale)
            y_offset = int(-self.config["lidar_y_min"] * y_scale)
            reduced_scale = self.config["anchor_bbox_feature"]["reduced_scale"]
            point_x = int((ref_bbox[0]*x_scale + x_offset)/reduced_scale)   # (0~ 700/4)
            point_y = int((ref_bbox[1]*y_scale + y_offset)/reduced_scale)  #(0 ~ 700/4)
            if point_x < 0 or point_x > H - 1 or point_y < 0 or point_y > W - 1:
                continue
            for x_int in range(self.config["positive_range"]):
                pos_x = point_x - int(self.config["positive_range"]/2) + x_int
                for y_int in range(self.config["positive_range"]):
                    pos_y = point_y - int(self.config["positive_range"]/2) + y_int
                    if pos_x < 0 or pos_x > H - 1 or pos_y < 0 or pos_y > W - 1:
                        continue
                    positive_position_list.append([pos_x, pos_y])
                    if self.regress_type==0:
                        positive_position_regress.append([pos_x, pos_y])
                        positive_position_idx[i].append(temp_cnt)
                        temp_cnt+=1
                    else:
                        if pos_x == point_x and pos_y == point_y:
                            positive_position_regress.append([pos_x, pos_y]) #중심만 추가하기
                            positive_position_idx[i].append(temp_cnt)
                            temp_cnt+=1
        #     sample_idx = np.random.choice(len(positive_position_list), np.max(sample_threshold, len(positive_position_list)), replace=False)
        np.random.shuffle(positive_position_list)
        if len(positive_position_list) > sample_threshold:
            positive_position_list = positive_position_list[:sample_threshold]
        return positive_position_idx, np.array(positive_position_regress), positive_position_list

    def getPositionOfNegative(self, anchor_bbox_feature, positive_position_list, sample_threshold = 128):
        _, C, H, W = anchor_bbox_feature.shape
        negative_position_list = []
        sample = 0
        while(1):
            x = np.random.randint(H)
            y = np.random.randint(W)
            if [x, y] in positive_position_list:
                continue
            else:
                negative_position_list.append([x, y])
                sample += 1
            if sample > sample_threshold:
                break
        return negative_position_list

    def getClassSum(self, positive_position_list, negative_position_list, predicted_class, loss_class):
        positive_size = len(positive_position_list)
        negative_size = len(negative_position_list)
        positive_label = torch.ones((positive_size), dtype=torch.long).cuda()
        negative_label = torch.zeros((negative_size), dtype=torch.long).cuda()
        negative_position = torch.tensor(negative_position_list, dtype=torch.long).cuda()
        positive_position = torch.tensor(positive_position_list, dtype=torch.long).cuda()
        c = predicted_class[:, negative_position[:,0], negative_position[:,1]].permute(1,0)
        if positive_size > 0:
            a = predicted_class[:, positive_position[:,0], positive_position[:,1]].permute(1,0)            
            loss_sum = loss_class(a, positive_label) + loss_class(c, negative_label)
        else:
            loss_sum = loss_class(c, negative_label)
        return loss_sum

    def LossReg(self, ref_box, pred_box, a_box):

        ### Rel coordinate 기준

        # ref_box : [7,]
        # pred_box : [N, 14, ]
        # anchor_box : [N, 2, 7]

        N, num_anchor, char = a_box.shape
        ref_box = ref_box.unsqueeze(0).unsqueeze(0).repeat(N,num_anchor,1)
        pred_box = pred_box.reshape(N,num_anchor,char)
        xy_ref_offset = (ref_box[:,:,:2]-a_box[:,:,:2])/torch.sqrt(torch.pow(a_box[:,:,3:4],2) + torch.pow(a_box[:,:,4:5],2))
        z_ref_offset = (ref_box[:,:,2:3]-a_box[:,:,2:3])/(a_box[:,:,5:6])
        whd_ref_offset = torch.log(ref_box[:,:,3:6]/(a_box[:,:,3:6]))
        ori_ref_offset = torch.atan2(torch.sin(ref_box[:,:,6] - a_box[:,:,-1]), torch.cos(ref_box[:,:,6] - a_box[:,:,-1]))
        ref_offset = torch.cat([xy_ref_offset, z_ref_offset, whd_ref_offset, ori_ref_offset.unsqueeze(-1)], dim=-1)  #[N,2,7]

        l1_loss = self.loss_regress(pred_box[:,:,:char], ref_offset)
    
        loss = 1.0/(N*num_anchor*char) * torch.sum(l1_loss)

        return loss

    def getRegSum(self, index, positive_position_list, reference_bboxes, predicted_regress_feature, anchor):

        # reference_bboxes = x : 0~70.0,  y : -35.0 ~ 35.0
        # positive_position_list = x : 700/4  y: 700/4 , size : [700/4, 700/4]
        # predicted_regress_feature = x : ?, y: ?, size : [700/4, 700/4]
        # anchor = x : 0~70.0, y : -35.0~35.0,  size : [700/4, 700/4]
        
        # back propagation is error. Batch sum and loss device type is error i think
        reg_loss = torch.zeros(1).cuda()
        for idx, reference_box in enumerate(reference_bboxes):
            positive_position = torch.tensor(positive_position_list[index[idx]], dtype=torch.long).cuda()
            predicted_box = predicted_regress_feature[:,positive_position[:,0], positive_position[:,1]].permute(1,0)
            anchor_box = anchor[:,:, positive_position[:,0], positive_position[:,1]].permute(2,0,1) #[2,7]
            if predicted_box.shape[0] == 0:
                continue
            # predicted_box = torch.stack(box_list, dim=0) #[N,14], predicted bbox set
            # anchor_box = torch.stack(abox_list, dim=0) #[N,2,7], anchor bbox set
            
            loss = self.LossReg(reference_box, predicted_box, anchor_box)
            reg_loss +=loss


        return reg_loss

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # print(anchor_set_1[0, 0, :, :])
    # print(anchor_set_1[0, 1, :, :])
    # save_image(anchor_set_1[0, 0, :, :]/70.0, 'anchor/x.png')
    # save_image(anchor_set_1[0, 1, :, :]/(70.0)+0.5, 'anchor/y.png')
#### Not use this function but will be use next time...
# def getIOUfeature(anchor_bbox_feature, ref_bboxes):
#     '''
#     caution! it will be extremely slow i guess... need to make faster algorithm
#     :param anchor_bbox_feature: reference bbox feature (7,H,W)
#     :param ref_bboxes: predicted bbox feature (7,H,W)
#     :return:IOU_feature
#     '''
#     anchor_bbox_feature = anchor_bbox_feature.cpu()
#     C, H, W = anchor_bbox_feature.shape
#     IOU_feature = torch.zeros((1, H, W))
#     for h in range(H):
#         for w in range(W):
#             for ref_bbox in ref_bboxes:
#                 anchor_bbox = anchor_bbox_feature[:, h, w]
#                 distance = torch.sqrt(torch.sum(torch.pow((anchor_bbox[:3] - ref_bbox[:3]),2)))
#                 if distance < 4.0:
#                     ref_bbox_corners = get_3d_box(ref_bbox[:3], ref_bbox[3:6], ref_bbox[-1])
#                     anchor_bbox_corners = get_3d_box(anchor_bbox[:3], anchor_bbox[3:6], anchor_bbox[-1])
#                     (IOU_3d, IOU_2d) = box3d_iou(ref_bbox_corners, anchor_bbox_corners)
#                     IOU_feature[:, h, w] = IOU_3d
#                 else:
#                     IOU_feature[:, h, w] = 0
#     return IOU_feature

# def getClassRefFromDist(anchor_bbox_feature, ref_bboxes):
#     '''
#     its extremely slow. how could we change this?
#     :param anchor_bbox_feature: reference bbox. list(bboxes)
#     :param ref_bboxes: predicted bbox feature. (7,H,W)
#     :return:class_feature. (2, H, W)
#     '''
#     C, H, W = anchor_bbox_feature.shape
#     class_feature = torch.zeros((2, H, W)).cuda()
#     anchor_bbox_feature = anchor_bbox_feature.cuda()
#     ref_bboxes = ref_bboxes.cuda()
#     for h in range(H):
#         print("doing at ",h, "in class ref.")
#         for w in range(W):
#             for ref_bbox in ref_bboxes:
#                 anchor_bbox = anchor_bbox_feature[:, h, w]
#                 # distance = torch.sqrt(torch.sum(torch.pow((anchor_bbox[:3] - ref_bbox[:3]),2)))
#                 # if class_feature[0, h, w] == 0 and class_feature[1, h, w] == 0:
#                 #     if distance < 1:
#                 #         class_feature[0, h, w] = 1
#                 #     elif distance > 4:
#                 #         class_feature[1, h, w] = 1
#     return class_feature

# def getClassRefFromDist_fast(anchor_bbox_feature, ref_bboxes):
#     '''
#     :param anchor_bbox_feature: reference bbox. list(bboxes)
#     :param ref_bboxes: predicted bbox feature. (7,H,W)
#     :return:class_feature. (2, H, W)
#     '''
#     C, H, W = anchor_bbox_feature.shape
#     class_feature_pos = torch.zeros((1, H, W)).cuda()
#     class_feature_neg = torch.ones((1, H, W)).cuda()
#     class_feature = torch.cat((class_feature_pos,class_feature_neg))
#     # anchor_bbox_feature = anchor_bbox_feature.cuda()
#     for ref_bbox in ref_bboxes:
#         point_x = int(ref_bbox[0]*10/4)
#         point_y = int((ref_bbox[1]*10 + 350)/4)
#         if point_x < 0 or point_x > int(700/4) - 1 or point_y < 0 or point_y > int(700/4) - 1:
#             continue
#         for x_int in range(5):
#             for y_int in range(5):
#                 class_feature[0, point_x - 2 + x_int, point_y - 2 + y_int] = 1
#                 class_feature[1, point_x - 2 + x_int, point_y - 2 + y_int] = 0
#         # if class_feature[0, h, w] == 0 and class_feature[1, h, w] == 0:
#         #     if distance < 1:
#         #         class_feature[0, h, w] = 1
#         #     elif distance > 4:
#         #         class_feature[1, h, w] = 1
#     return class_feature