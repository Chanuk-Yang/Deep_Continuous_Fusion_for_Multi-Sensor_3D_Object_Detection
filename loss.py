import torch
import torch.nn as nn
import numpy as np
import random


from IOU import get_3d_box, box3d_iou
from data_import import arangeLabelData

def getAnchorboundingboxFeature():
    f_height = int(700/4) -1
    f_width = int(700/4) - 1
    width = 2.0
    length = 4.0
    height = 1.5
    anc_x = torch.matmul(
        torch.ones(f_height, 1), torch.linspace(-1.0, 1.0, f_width).view(1, f_width)).view(1, f_height, f_width)
    anc_y = torch.matmul(
        torch.linspace(-1.0, 1.0, f_height).view(f_height, 1), torch.ones(1, f_width)).view(1, f_height, f_width)
    anc_z = torch.ones(1, f_height, f_width) * 1
    anc_w = torch.ones(1, f_height, f_width) * width
    anc_l = torch.ones(1, f_height, f_width) * length
    anc_h = torch.ones(1, f_height, f_width) * height
    anc_ori = torch.ones(1, f_height, f_width) * 0
    anc_ori_90 = torch.ones(1, f_height, f_width) * 3.1415926/2
    anc_set_1 = torch.cat((anc_x, anc_y, anc_z, anc_l, anc_w, anc_h, anc_ori), 0)
    anc_set_2 = torch.cat((anc_x, anc_y, anc_z, anc_l, anc_w, anc_h, anc_ori_90), 0)
    return anc_set_1, anc_set_2

def getPositionOfPositive(anchor_bbox_feature, ref_bboxes, sample_threshold = 128):
    C, H, W = anchor_bbox_feature.shape
    positive_position_list = []
    for ref_bbox in ref_bboxes:
        point_x = int(ref_bbox[0]*10/4)
        point_y = int((ref_bbox[1]*10 + 350)/4)
        if point_x < 0 or point_x > H - 1 or point_y < 0 or point_y > W - 1:
            continue
        for x_int in range(5):
            pos_x = point_x - 2 + x_int
            for y_int in range(5):
                pos_y = point_y - 2 + y_int
                if pos_x < 0 or pos_x > H - 1 or pos_y < 0 or pos_y > W - 1:
                    continue
                positive_position_list.append([pos_x, pos_y])
    random.shuffle(positive_position_list)
    if len(positive_position_list) > sample_threshold:
        positive_position_list = positive_position_list[:sample_threshold]
    return positive_position_list


def getPositionOfNegative(anchor_bbox_feature, positive_position_list, sample_threshold = 128):
    C, H, W = anchor_bbox_feature.shape
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


def getOffset(target, anchors):
    # target: (N,7,H,W), 7 means xyz pos, whd size, orientation
    # anchors: tuple of (N,7,H,W)
    # total_offset: returns of list of offset
    total_offset =[]
    for anchor in anchors:
        # anchor = torch.zeros(target.size())
        offset_xyz = (target[:3] - anchor[:3]) / anchor[:3]
        offset_whd = torch.log(target[3:6]/anchor[3:6])
        offset_ori = target[-1] - anchor[-1]
        offset = torch.cat((offset_xyz, offset_whd, offset_ori), 0)
        total_offset.append(offset)
    return total_offset


def getClassSum(positive_position_list, negative_position_list, predicted_class, loss_class):
    N, H, W = predicted_class.shape
    positive_list = []
    negative_list = []
    positive_size = len(positive_position_list)
    negative_size = len(negative_position_list)
    positive_label = torch.ones(positive_size, dtype=torch.long)
    negative_label = torch.zeros(negative_size, dtype=torch.long)

    for negative_position in negative_position_list:
        sampled_feature_negative = predicted_class[:, negative_position[0], negative_position[1]]
        negative_list.append(sampled_feature_negative)
    c = torch.stack(negative_list, 0)
    if positive_size > 0:
        for positive_position in positive_position_list:
            sampled_feature_positive = predicted_class[:, positive_position[0], positive_position[1]]
            positive_list.append(sampled_feature_positive)
        a = torch.stack(positive_list, 0)
        loss_sum = loss_class(a.cpu(), positive_label.cpu()) + loss_class(c.cpu(), negative_label.cpu())
    else:
        loss_sum = loss_class(c.cpu(), negative_label.cpu())
    return loss_sum


class LossClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, predicted_feature, binary_label):
        loss = self.loss(predicted_feature, binary_label)
        return loss


class LossTotal(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_class = LossClass()
        self.i = 0
        anchor_set_1, anchor_set_2 = getAnchorboundingboxFeature()
        self.anchor_set = torch.cat((anchor_set_1,anchor_set_2), dim = 0)

    def forward(self, reference_bboxes, predicted_class):
        positive_position_list = getPositionOfPositive(self.anchor_set, torch.tensor(reference_bboxes))
        negative_position_list = getPositionOfNegative(self.anchor_set, positive_position_list)
        total_loss_class = getClassSum(positive_position_list, negative_position_list, predicted_class[0,:2,:,:], self.loss_class)
        total_loss_class += getClassSum(positive_position_list, negative_position_list, predicted_class[0,2:4,:,:], self.loss_class)
        """
        TODO
        1. make regression loss
        2. omit the reference object when the points are few in bounding box (or...)
        
        """


        total_loss = total_loss_class
        return total_loss


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