import torch
import torch.nn as nn
from torchvision.utils import save_image

import numpy as np
import random
import torch.nn.functional as F
import os

from IOU import get_3d_box, box3d_iou

def getAnchorboundingboxFeature():
    f_height = int(700/4) 
    f_width = int(700/4) 
    width = 2.0
    length = 4.0
    height = 1.5
    anc_x = torch.matmul(
        torch.linspace(0, 70.0, f_height).view(f_height, 1), torch.ones(1, f_width)).view(1, f_height, f_width)
    anc_y = torch.matmul(
        torch.ones(f_height, 1), torch.linspace(-35.0, 35.0, f_width).view(1, f_width)).view(1, f_height, f_width)
    anc_z = torch.ones(1, f_height, f_width) * (-4.5) # need to modify!
    anc_w = torch.ones(1, f_height, f_width) * width
    anc_l = torch.ones(1, f_height, f_width) * length
    anc_h = torch.ones(1, f_height, f_width) * height
    anc_ori = torch.ones(1, f_height, f_width) * 0
    anc_ori_90 = torch.ones(1, f_height, f_width) * 3.1415926/2
    anc_set_1 = torch.cat((anc_x, anc_y, anc_z, anc_l, anc_w, anc_h, anc_ori), 0).unsqueeze(0).cuda()
    anc_set_2 = torch.cat((anc_x, anc_y, anc_z, anc_l, anc_w, anc_h, anc_ori_90), 0).unsqueeze(0).cuda()
    return anc_set_1, anc_set_2

def getPositionOfPositive(anchor_bbox_feature, ref_bboxes, regress_type, sample_threshold = 128):
    _, C, H, W = anchor_bbox_feature.shape
    positive_position_list = []
    positive_position_regress = []
    positive_position_idx = {}
    temp_cnt = 0
    for i, ref_bbox in enumerate(ref_bboxes):
        positive_position_idx[i] = []
        point_x = int(ref_bbox[0]*10/4)   # (0~ 700/4)
        point_y = int((ref_bbox[1]*10 + 350)/4)  #(0 ~ 700/4)
        if point_x < 0 or point_x > H - 1 or point_y < 0 or point_y > W - 1:
            continue
        for x_int in range(5):
            pos_x = point_x - 2 + x_int
            for y_int in range(5):
                pos_y = point_y - 2 + y_int
                if pos_x < 0 or pos_x > H - 1 or pos_y < 0 or pos_y > W - 1:
                    continue
                positive_position_list.append([pos_x, pos_y])
                if regress_type==0:
                    positive_position_regress.append([pos_x, pos_y])
                    positive_position_idx[i].append(temp_cnt)
                    temp_cnt+=1
                else:
                    if x_int==2 and y_int==2:
                        positive_position_regress.append([pos_x, pos_y]) #중심만 추가하기
                        positive_position_idx[i].append(temp_cnt)
                        temp_cnt+=1
#     sample_idx = np.random.choice(len(positive_position_list), np.max(sample_threshold, len(positive_position_list)), replace=False)
    np.random.shuffle(positive_position_list)
    if len(positive_position_list) > sample_threshold:
        positive_position_list = positive_position_list[:sample_threshold]
    return positive_position_idx, np.array(positive_position_regress), positive_position_list

def getPositionOfNegative(anchor_bbox_feature, positive_position_list, sample_threshold = 128):
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


def LossReg(ref_box, pred_box, a_box):

    ### Rel coordinate 기준

    # ref_box : [7,]
    # pred_box : [N, 14, ]
    # anchor_box : [N, 2, 7]

    N, num_anchor, char = a_box.shape
    ref_box = ref_box.unsqueeze(0).unsqueeze(0).repeat(N,num_anchor,1)
    pred_box = pred_box.reshape(N,num_anchor,char)
    xyz_ref_offset = (ref_box[:,:,:3]-a_box[:,:,:3])/(a_box[:,:,3:6])
    whd_ref_offset = torch.log(ref_box[:,:,3:6]/(a_box[:,:,3:6]))
    ori_ref_offset = ref_box[:,:,6] - a_box[:,:,-1]
    ref_offset = torch.cat([xyz_ref_offset, whd_ref_offset, ori_ref_offset.unsqueeze(-1)], dim=-1)

    l1_loss = F.smooth_l1_loss(pred_box, ref_offset, reduce=False)
    # print(l1_loss[:,0,:])
    # xyz_error = pred_box[:,:,:3] - xyz_ref_offset
    # whd_error = pred_box[:,:,3:6] - whd_ref_offset
    # ori_error = pred_box[:,:,-1] - ori_ref_offset
    # total_error = torch.cat([xyz_error, whd_error,ori_error.unsqueeze(-1)], dim=-1) #[N,2,7]
    # l1_loss = F.smooth_l1_loss(total_error, torch.zeros_like(total_error).cuda()) #[N,2,7]
    
    loss = 1.0/(N*num_anchor*char) * torch.sum(l1_loss)
    # if loss.item() > 0.2:
    #     print("l1loss: ",l1_loss)
    return loss



def getRegSum(index, positive_position_list, reference_bboxes, predicted_regress_feature, anchor, pred_bbox_f):

    # reference_bboxes = x : 0~70.0,  y : -35.0 ~ 35.0
    # positive_position_list = x : 700/4  y: 700/4 , size : [700/4, 700/4]
    # predicted_regress_feature = x : ?, y: ?, size : [700/4, 700/4]
    # anchor = x : 0~70.0, y : -35.0~35.0,  size : [700/4, 700/4]
    
    # back propagation is error. Batch sum and loss device type is error i think
    
    positive_size = len(positive_position_list)
    reg_loss = torch.zeros(1)
    for idx, reference_box in enumerate(reference_bboxes):

        box_list = []
        abox_list = []
        bbox_real_list = []
        for positive_position in positive_position_list[index[idx]]:  ### for loop reduction ###
            box_list.append(predicted_regress_feature[:,positive_position[0], positive_position[1]])
            abox_list.append(anchor[:,:, positive_position[0], positive_position[1]]) #[2,7]
            bbox_real_list.append(pred_bbox_f[:, positive_position[0], positive_position[1]])

        if len(box_list) == 0:
            continue
        predicted_box = torch.stack(box_list, dim=0) #[N,14], offset bbox set
        anchor_box = torch.stack(abox_list, dim=0) #[N,2,7], original bbox set
        N, num_anchor, char = anchor_box.shape
        bbox_real = torch.stack(bbox_real_list, dim=0).reshape(N,num_anchor,char)
        
        loss = LossReg(reference_box, predicted_box, anchor_box).cpu()
        # if loss.item() > 0.2:
        #     N, num_anchor, char = anchor_box.shape
        #     reference_box_ = reference_box[:7].unsqueeze(0).unsqueeze(0).repeat(N,num_anchor,1)
        #     difference = bbox_real-reference_box_.cpu()
        #     print ("bbox_real_list: ", bbox_real)
        #     print ("difference: ", difference)
        #     print ("ref_bbox: ", reference_box)
        #     print ("loss: ", loss)
        reg_loss +=loss


    return reg_loss



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

    def forward(self, reference_bboxes_batch, num_ref_bbox_batch, predicted_class_feature_batch, predicted_regress_feature_batch, pred_bbox_f, regress_type=0):
        # reference_bboxes : B, max_n(20), 8
        B, max_num, _ = reference_bboxes_batch.shape
        
        # print('reference_bboxes_batch size : ',reference_bboxes_batch.shape)
        
        # print('num_ref_bbox_batch size : ',num_ref_bbox_batch.shape)
        
        # print('predicted_class_feature_batch size : ',predicted_class_feature_batch.shape)
        
        # print('predicted_regress_feature_batch size : ',predicted_regress_feature_batch.shape)
        
        total_loss = torch.zeros(1)
        
        # print('anchor_set size before : ',self.anchor_set.shape)
        self.anchor_set_ = self.anchor_set.unsqueeze(0).repeat(B,1,1,1,1)
        
        # print('anchor_set size : ',self.anchor_set_.shape)
        for b in range(B):
            reference_bboxes = reference_bboxes_batch[b,:num_ref_bbox_batch[b]]
            predicted_class_feature = predicted_class_feature_batch[b]
            predicted_regress_feature = predicted_regress_feature_batch[b]
            anchor = self.anchor_set_[b]
            
            
            IDX, positive_position_list_all, positive_position_list = getPositionOfPositive(anchor,reference_bboxes, regress_type)

            negative_position_list = getPositionOfNegative(anchor, positive_position_list)
            
            total_loss_class = getClassSum(positive_position_list, negative_position_list, predicted_class_feature[:2,:,:], self.loss_class)    # per anchor
            total_loss_class += getClassSum(positive_position_list, negative_position_list, predicted_class_feature[2:4,:,:], self.loss_class)  # per anchor

            ## anchor의 좌표를 절대 좌표로 바꿔야 한다.
            ## position_list_all은 positive_position이 다 들어간것, true 하나당 한 픽셀이려면 수정 필요

            Reg_loss = getRegSum(IDX, positive_position_list_all, reference_bboxes, predicted_regress_feature, anchor,pred_bbox_f[b])

            """
            TODO
            ## prerequsite
            ##   anchorbox 위치, 높이, 거리 등 -> 하나의 point당 2개씩인가??
            ##   true box 위치 높이 거리 등
            ##   pred box 위치 높이 거리 등
            ## 1. true anchor box의 offset을 구하기
            ## 2. pred anchor box의 offset을 구하기
            1. make regression loss
            2. omit the reference object when the points are few in bounding box (or...)
            """
            total_loss += total_loss_class + 10 * Reg_loss

        return total_loss
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    anchor_set_1, anchor_set_2 = getAnchorboundingboxFeature()
    print(anchor_set_1[0, 0, :, :])
    print(anchor_set_1[0, 1, :, :])
    save_image(anchor_set_1[0, 0, :, :]/70.0, 'anchor/x.png')
    save_image(anchor_set_1[0, 1, :, :]/(70.0)+0.5, 'anchor/y.png')
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