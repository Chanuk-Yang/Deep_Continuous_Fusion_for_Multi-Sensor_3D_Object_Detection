# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
from torchvision import models


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        if self.should_apply_shortcut:
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1), bias=False)
        else:
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if self.should_apply_shortcut:
            self.down_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), stride=(2, 2),
                                       bias=False)
            self.down_bn = nn.BatchNorm2d(self.out_channels, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)

    def forward(self, x):
        if self.should_apply_shortcut:
            residual = self.down_bn(self.down_conv(x))
        else:
            residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class Resnet18Customed(nn.Module):
    def __init__(self, out_feature=(64, 128, 192, 256)):
        super(Resnet18Customed, self).__init__()
        self.conv1 = nn.Conv2d(32, out_feature[0], kernel_size=(7, 7), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(ResidualBlock(out_feature[0], out_feature[0]))
        self.layer2 = nn.Sequential(ResidualBlock(out_feature[0], out_feature[1]),
                                    ResidualBlock(out_feature[1], out_feature[1]))
        self.layer3 = nn.Sequential(ResidualBlock(out_feature[1], out_feature[2]),
                                    ResidualBlock(out_feature[2], out_feature[2]),
                                    ResidualBlock(out_feature[2], out_feature[2]))
        self.layer4 = nn.Sequential(ResidualBlock(out_feature[2], out_feature[3]),
                                    ResidualBlock(out_feature[3], out_feature[3]),
                                    ResidualBlock(out_feature[3], out_feature[3]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# TO BE CONTINUE...................................
# class NonMaximumSuppression(nn.Module):
#     def __init__(self,IOU_thres):
#         self.IOU_thres = IOU_thres
#
#     def forward(self, x_cls, x_reg, score_sampling = 100):  # Caution! put features that corresponding to ONE anchor
#         x_cls_pos = x_cls[0].view(-1)
#         x_cls_neg = x_cls[1].view(-1)
#         sorted_pos, indice_pos = torch.sort(x_cls_pos, descending=True)
#         top_indice = indice_pos[:score_sampling]
#         sorted_neg, indice_neg = torch.sort(x_cls_neg, descending=True)
#         bot_indice = indice_neg[:score_sampling]
#         for i in range(score_sampling):
#             if i == 0:
#                 bounding_box = x_reg[top_indice[i]//640][top_indice[i]%640].view(1,7)
#             else:
# ===================================================================

class OffsettoBbox(nn.Module):
    def __init__(self):
        super(OffsettoBbox, self).__init__()
        f_height = int(700/4) -1
        f_width = int(700/4) - 1
        width = 2.0
        length = 4.0
        height = 1.5

        anc_x = torch.matmul(
            torch.ones(f_height, 1), torch.linspace(0.0, 70.0, f_width).view(1, f_width)).view(1, f_height, f_width)
        anc_y = torch.matmul(
            torch.linspace(-35.0, 35.0, f_height).view(f_height, 1), torch.ones(1, f_width)).view(1, f_height, f_width)
        anc_x = torch.matmul(
            torch.ones(f_height, 1), torch.linspace(0.0, 70.0, f_width).view(1, f_width)).view(1, f_height, f_width)
        anc_y = torch.matmul(
            torch.linspace(-35.0, 35.0, f_height).view(f_height, 1), torch.ones(1, f_width)).view(1, f_height, f_width)
        anc_z = torch.ones(1, f_height, f_width) * 1
        anc_w = torch.ones(1, f_height, f_width) * width
        anc_l = torch.ones(1, f_height, f_width) * length
        anc_h = torch.ones(1, f_height, f_width) * height
        anc_ori = torch.ones(1, f_height, f_width) * 0
        anc_ori_90 = torch.ones(1, f_height, f_width) * 3.1415926/2
        anc_set_1 = torch.cat((anc_x, anc_y, anc_z, anc_l, anc_w, anc_h, anc_ori), 0).cuda()
        anc_set_2 = torch.cat((anc_x, anc_y, anc_z, anc_l, anc_w, anc_h, anc_ori_90), 0).cuda()
        self.anc_set = torch.cat((anc_set_1,anc_set_2),0).unsqueeze(0)

    def forward(self, x):
        """
        x: x_reg [b,num_anc*7,wid,hei]
        """

        pred_xyz_1 = x[:,:3,:,:] * self.anc_set[:,:3,:,:] + self.anc_set[:,:3,:,:]
        pred_whl_1 = torch.exp(x[:,3:6,:,:]) * self.anc_set[:,3:6,:,:]
        pred_ori_1 = x[:,6:7,:,:] + self.anc_set[:,6:7,:,:]

        pred_xyz_2 = x[:,7:10,:,:] * self.anc_set[:,7:10,:,:] + self.anc_set[:,7:10,:,:]
        pred_whl_2 = torch.exp(x[:,10:13,:,:]) * self.anc_set[:,10:13,:,:]
        pred_ori_2 = x[:,13:14,:,:] + self.anc_set[:,13:14,:,:]

        pred_bbox_feature = torch.cat((pred_xyz_1, pred_whl_1, pred_ori_1,
                                      pred_xyz_2, pred_whl_2, pred_ori_2), dim=1)

        return pred_bbox_feature


class LidarBackboneNetwork(nn.Module):
    def __init__(self, out_feature=(64, 128, 192, 256),Num_anchor = 2):
        super(LidarBackboneNetwork, self).__init__()
        self.backbone = Resnet18Customed(out_feature)
        self.conv1 = nn.Conv2d(out_feature[-1], out_feature[-2], kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)
        self.classconv = nn.Conv2d(out_feature[-2], Num_anchor*2, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bbox3dconv = nn.Conv2d(out_feature[-2], Num_anchor*7, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.score_threshold = 0.7
        self.IOU_threshold = 0.7

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.upscale(x)
        x_cls = self.classconv(x)
        x_reg = self.bbox3dconv(x)
        return x_cls, x_reg


class ObjectDetection_DCF(nn.Module):
    def __init__(self):
        super(ObjectDetection_DCF, self).__init__()
        self.offset_to_bbox = OffsettoBbox()
        self.lidar_backbone = LidarBackboneNetwork()
        self.image_backbone = models.resnet18(pretrained=True)

    def forward(self, x_lidar, x_image):
        lidar_pred_cls, lidar_pred_reg = self.lidar_backbone(x_lidar)
        image_ = self.image_backbone(x_image)
        lidar_pred_bbox = self.offset_to_bbox(lidar_pred_reg)
        """
        TODO
        1. make continuous fusion layer from image
        2. add with lidar feature
        """
        return lidar_pred_cls, lidar_pred_reg, lidar_pred_bbox


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_backbone = models.resnet18(pretrained=True)
    model = LidarBackboneNetwork()
    pred = model(torch.ones(4, 32, 480, 640))
    pred2 = image_backbone(torch.ones(4, 3, 480, 640))
    a = 1
    print("model inference is good")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
