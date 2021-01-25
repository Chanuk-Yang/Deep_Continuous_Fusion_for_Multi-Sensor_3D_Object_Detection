# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
from torchvision import models

class AnchorBondingBoxFeature(nn.Module):
    def __init__(self):
        super(AnchorBondingBoxFeature,self).__init__()
        self.f_height = int(700/4)
        self.f_width = int(700/4)
        self.width = 2.0
        self.length = 4.0
        self.height = 1.5
        
    def forward(self):
        anc_x = torch.matmul(
                torch.linspace(0, 70.0, self.f_height).view(self.f_height, 1), torch.ones(1, self.f_width)).view(1, self.f_height, self.f_width)
        anc_y = torch.matmul(
                torch.ones(self.f_height, 1), torch.linspace(-35.0, 35.0, self.f_width).view(1, self.f_width)).view(1, self.f_height, self.f_width)
        anc_z = torch.ones(1, self.f_height, self.f_width) * (-4.5)
        anc_w = torch.ones(1, self.f_height, self.f_width) * self.width
        anc_l = torch.ones(1, self.f_height, self.f_width) * self.length
        anc_h = torch.ones(1, self.f_height, self.f_width) * self.height 
        anc_ori = torch.ones(1, self.f_height, self.f_width) * 0
        anc_ori_90 = torch.ones(1, self.f_height, self.f_width) * 3.1415926/2
        anc_set_1 = torch.cat((anc_x, anc_y, anc_z, anc_l, anc_w, anc_h, anc_ori), 0)
        anc_set_2 = torch.cat((anc_x, anc_y, anc_z, anc_l, anc_w, anc_h, anc_ori_90), 0)
        anc_set = torch.cat((anc_set_1,anc_set_2), dim=0).unsqueeze(0) # dim = [1, 2*7, self.f_height, self.f_width]
        return anc_set

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock,self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        if self.should_apply_shortcut:
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1), bias=False)
        else:
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
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
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu2(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResnetCustomed(nn.Module):
    def __init__(self, out_feature=(32, 64, 128, 192, 256)):
        super(ResnetCustomed, self).__init__()
        self.layer1 = nn.Sequential(ResidualBlock(out_feature[0], out_feature[0]))
        self.layer2 = nn.Sequential(ResidualBlock(out_feature[0], out_feature[1]),
                                    ResidualBlock(out_feature[1], out_feature[1]))
        self.layer3 = nn.Sequential(ResidualBlock(out_feature[1], out_feature[2]),
                                    ResidualBlock(out_feature[2], out_feature[2]),
                                    ResidualBlock(out_feature[2], out_feature[2]),
                                    ResidualBlock(out_feature[2], out_feature[2]))
        self.layer4 = nn.Sequential(ResidualBlock(out_feature[2], out_feature[3]),
                                    ResidualBlock(out_feature[3], out_feature[3]),
                                    ResidualBlock(out_feature[3], out_feature[3]),
                                    ResidualBlock(out_feature[3], out_feature[3]),
                                    ResidualBlock(out_feature[3], out_feature[3]),
                                    ResidualBlock(out_feature[3], out_feature[3]))
        self.layer5 = nn.Sequential(ResidualBlock(out_feature[3], out_feature[4]),
                                    ResidualBlock(out_feature[4], out_feature[4]),
                                    ResidualBlock(out_feature[4], out_feature[4]),
                                    ResidualBlock(out_feature[4], out_feature[4]),
                                    ResidualBlock(out_feature[4], out_feature[4]),
                                    ResidualBlock(out_feature[4], out_feature[4]))

    def forward(self, x):
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        x4 = self.layer5(x3)
        return x4, x3, x2


class OffsettoBbox(nn.Module):
    def __init__(self):
        super(OffsettoBbox, self).__init__()
        self.anchor_bbox_feature = AnchorBondingBoxFeature()
        
    def forward(self, x):
        """
        x: x_reg [b,num_anc*7,wid,hei]
        """
        anc_set = self.anchor_bbox_feature().cuda()
        pred_xy_1 = x[:,:2,:,:] * torch.sqrt(torch.pow(anc_set[:,3:4,:,:],2) + torch.pow(anc_set[:,4:5,:,:],2)) + anc_set[:,:2,:,:]
        pred_z_1 = x[:,2:3,:,:] * (anc_set[:,5:6,:,:]) + anc_set[:,2:3,:,:]

        pred_whl_1 = torch.exp(x[:,3:6,:,:]) * anc_set[:,3:6,:,:]
        pred_ori_1 = torch.atan2(torch.sin(x[:,6:7,:,:] + anc_set[:,6:7,:,:]), torch.cos(x[:,6:7,:,:] + anc_set[:,6:7,:,:]))
        pred_xy_2 = x[:,7:9,:,:] * torch.sqrt(torch.pow(anc_set[:,10:11,:,:],2) + torch.pow(anc_set[:,11:12,:,:],2)) + anc_set[:,7:9,:,:]
        pred_z_2 = x[:,9:10,:,:] * (anc_set[:,12:13,:,:]) + anc_set[:,9:10,:,:]
        pred_whl_2 = torch.exp(x[:,10:13,:,:]) * anc_set[:,10:13,:,:]
        pred_ori_2 = torch.atan2(torch.sin(x[:,13:14,:,:] + anc_set[:,13:14,:,:]), torch.cos(x[:,13:14,:,:] + anc_set[:,13:14,:,:]))
        pred_bbox_feature = torch.cat((pred_xy_1, pred_z_1, pred_whl_1, pred_ori_1,
                                      pred_xy_2, pred_z_2, pred_whl_2, pred_ori_2), dim=1)
        return pred_bbox_feature
        

class LidarBackboneNetwork(nn.Module):
    def __init__(self, out_feature=(32, 64, 128, 192, 256),Num_anchor = 2):
        super(LidarBackboneNetwork, self).__init__()
        self.backbone = ResnetCustomed(out_feature)
        self.num_anchor = Num_anchor
        
        # FPN
        self.latconv1 = nn.Conv2d(out_feature[-2], out_feature[-2], kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.downconv1 = nn.Conv2d(out_feature[-1], out_feature[-2], kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.upscale1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.latconv2 = nn.Conv2d(out_feature[-3], out_feature[-2], kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.upscale2 = nn.UpsamplingBilinear2d(size=175) # NEED TO GENERALIZE IN BEV SIZE
        self.conv3 = nn.Conv2d(out_feature[-2], out_feature[-2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        self.classconv = nn.Conv2d(out_feature[-2], Num_anchor*2, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        self.bbox3dconv = nn.Conv2d(out_feature[-2], Num_anchor*7, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):
        x4, x3, x2 = self.backbone(x)
        x3 = self.latconv1(x3)
        x3_ = self.upscale1(self.downconv1(x4))
        x3 += x3_
        x2 = self.latconv2(x2)
        x2_ = self.upscale2(x3)
        x2 += x2_
        x_pred = self.conv3(x2)
        x_cls = self.classconv(x_pred)
        x_cls_1 = self.softmax1(x_cls[:,:2])
        x_cls_2 = self.softmax2(x_cls[:,2:4])
        x_cls = torch.cat((x_cls_1,x_cls_2), dim=1)
        x_reg = self.bbox3dconv(x_pred)
        return x_cls, x_reg


class ObjectDetection_DCF(nn.Module):
    def __init__(self):
        super(ObjectDetection_DCF, self).__init__()
        self.offset_to_bbox = OffsettoBbox()
        self.lidar_backbone = LidarBackboneNetwork()
        # self.image_backbone = models.resnet18(pretrained=True)

    def forward(self, x_lidar, x_image):
        lidar_pred_cls, lidar_pred_reg = self.lidar_backbone(x_lidar)
        # image_ = self.image_backbone(x_image)
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
    pred = model(torch.ones(4, 32, 700, 700))
    print(pred[0].shape)
    pred2 = image_backbone(torch.ones(4, 3, 480, 640))
    a = 1
    print("model inference is good")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
