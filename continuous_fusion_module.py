import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import quaternion
import os
from data_import_carla import CarlaDataset


class MLP(nn.Module):
    def __init__(self,out_size):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(out_size[0], out_size[1])
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(out_size[1], out_size[2])
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(out_size[2], out_size[3])
        self.relu_3 = nn.ReLU()
        self.linear_4 = nn.Linear(out_size[3], out_size[4])
        self.relu_4 = nn.ReLU()
        self.linear_5 = nn.Linear(out_size[4], 1)
        self.relu_5 = nn.ReLU()

    def forward(self,x):
        x = self.relu_1(self.linear_1(x))
        x = self.relu_2(self.linear_2(x))
        x = self.relu_3(self.linear_3(x))
        x = self.relu_4(self.linear_4(x))
        x = self.relu_5(self.linear_5(x))
        return x

class BilinearResampling(nn.Module):
    def __init__(self, downscale=1):
        super(BilinearResampling, self).__init__()
        self.downscale = downscale

    def forward(self, feature_map, target_uv,):
        u = target_uv[:,0].view(-1)/self.downscale
        v = target_uv[:,1].view(-1)/self.downscale

        u_lower = u.type(torch.long)
        v_lower = v.type(torch.long)
        u_upper = u_lower + 1
        v_upper = v_lower + 1
        du = u - u_lower.type(torch.float)
        dv = v - v_lower.type(torch.float)
        pixels_out = feature_map[:,v_lower,u_lower]*(1 - dv)*(1 - du) + \
                    feature_map[:,v_upper,u_lower]*dv*(1 - du) + \
                    feature_map[:,v_lower,u_upper]*(1 - dv)*du + \
                    feature_map[:,v_upper,u_upper]*dv*du
        return pixels_out

class ContinuousFusion(nn.Module):
    def __init__(self, BEV_features_shape_set, image_features_shape_set):
        super(ContinuousFusion, self).__init__()
        out_size = (963, 128, 128, 128, 128)
        self.mlp = {}
        self.feature_3d_grid_z = {}
        self.feature_2d_grid = {}
        self.bev_feature_shape_set = {}

        # i 대신 다른방법 찾는것도 필요할듯...
        i = 0
        for BEV_feature_shape in BEV_features_shape_set:
            self.batch, cha, col, row = BEV_feature_shape
            self.bev_feature_shape_set[i] = (cha, col, row)
            feature_3d_grid_x = torch.linspace(0, 70.0, col).unsqueeze(0).permute(1,0).repeat(1,row).unsqueeze(0).repeat(cha,1,1).cuda()
            feature_3d_grid_y = torch.linspace(-35.0, 35.0, row).unsqueeze(0).repeat(col,1).unsqueeze(0).repeat(cha,1,1).cuda()
            self.feature_3d_grid_z[i] = torch.linspace(0, 3.2, cha).unsqueeze(1).unsqueeze(1).repeat(1,col,row).cuda()
            self.feature_2d_grid[i] = torch.cat((feature_3d_grid_x[0].unsqueeze(-1),feature_3d_grid_y[0].unsqueeze(-1)), dim=-1).view(-1,2)
            self.mlp[i] = MLP(out_size).cuda()
            i+=1
        self.bev_feature_set_num = i

    def bilinear_resampling(self, feature_map, target_uv, downscale=1):
        """
        feature_map: image feature map (C, H/downscale, W/downscale)
        target_uv: location of raw image, not feature map (N,2)
        downscale: ratio of image / feature_map

        """
        u = target_uv[:,0].view(-1)/downscale
        v = target_uv[:,1].view(-1)/downscale

        u_lower = u.type(torch.long)
        v_lower = v.type(torch.long)
        u_upper = u_lower + 1
        v_upper = v_lower + 1
        du = u - u_lower.type(torch.float)
        dv = v - v_lower.type(torch.float)
        pixels_out = feature_map[:,v_lower,u_lower]*(1 - dv)*(1 - du) + \
                    feature_map[:,v_upper,u_lower]*dv*(1 - du) + \
                    feature_map[:,v_lower,u_upper]*(1 - dv)*du + \
                    feature_map[:,v_upper,u_upper]*dv*du
        return pixels_out

    def forward(self, BEV_features_set, image_features_set, pointcloud_raw_, uv_, num_points):
        """
        BEV_features_set: tuple of BEV features. (feature1, feature2, feature3, feature4)
        image_features_set: tuple of Image features. it should be downscale twice sequencially (feature1, feature2, feature3, feature4)
        pointcloud_raw: bunch of points inner both BEV and Image. shape:(B, N, xyz)
        uv : image location of pointcloud_raw. shape:(B, N, uv)
        num_points: number of points with all batches. shape:(B)

        """
        for b in range(self.batch):
            pointcloud_raw = pointcloud_raw_[b,:num_points[b],:]
            uv = uv_[b,:num_points[b],:]

            # Step1, 2, 3: KNN search and Project to Camera View
            for i in range(self.bev_feature_set_num):
                dist_xy_all = torch.norm(pointcloud_raw[:,:2] - self.feature_2d_grid[i].unsqueeze(1), dim=-1)
                knn = dist_xy_all.topk(1)
                feature_2d_grid_nearest_point = pointcloud_raw[knn.indices.view(-1)]
                feature_2d_grid_nearest_uv = uv[knn.indices.view(-1)]

                # Step4: Retrieve Image + Geometric Features
                pixels_out_one = bilinear_resampling(image_features_set[0][b], feature_2d_grid_nearest_uv, downscale=4)
                pixels_out_two = bilinear_resampling(image_features_set[1][b], feature_2d_grid_nearest_uv, downscale=8)
                pixels_out_three = bilinear_resampling(image_features_set[2][b], feature_2d_grid_nearest_uv, downscale=16)
                pixels_out_four = bilinear_resampling(image_features_set[3][b], feature_2d_grid_nearest_uv, downscale=32)
                mlp_in_ = torch.cat((pixels_out_one, pixels_out_two, pixels_out_three, pixels_out_four), dim=0)
                for z_ind in range(self.bev_feature_shape_set[i][0]):
                    feature_2d_grid_z = self.feature_3d_grid_z[i][z_ind].view(-1).unsqueeze(-1)
                    offsets = feature_2d_grid_nearest_point - torch.cat((self.feature_2d_grid[i], feature_2d_grid_z), dim=-1)
                    mlp_in = torch.cat((mlp_in_, offsets.permute(1,0)), dim=0)
                    
                    # Step5: Output features to target pixel
                    mlp_out = self.mlp[i](mlp_in.permute(1,0))
                    cha, col, row = self.bev_feature_shape_set[i]
                    BEV_features_set[i][b,z_ind] += mlp_out.view(col,row)
        return BEV_features_set


if __name__ == '__main__':
    # image_backbone = models.resnet18(pretrained=True)
    # pred = image_backbone(torch.ones(1, 3, 480, 640))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    batch = 4
    dataset = CarlaDataset(mode="test")
    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch,
                                          shuffle=True)

    image_feature_one = torch.ones(batch,64,120,160).cuda() # input
    image_feature_two = torch.ones(batch,128,60,80).cuda()
    image_feature_three = torch.ones(batch,256,30,40).cuda()
    image_feature_four = torch.ones(batch,512,15,20).cuda()
    
    bev_feature_one = torch.ones([batch,32,350,350]).cuda() # input
    bev_feature_two = torch.ones([batch,64,175,175]).cuda()
    bev_feature_three = torch.ones([batch,64,50,50]).cuda()
    bev_feature_four = torch.ones([batch,128,50,50]).cuda()

    tuple_of_image_features = (image_feature_one, image_feature_two, image_feature_three, image_feature_four)
    list_of_bev_features = [bev_feature_one, bev_feature_two, bev_feature_three, bev_feature_four]

    BEV_features_shape_set = [bev_feature.shape for bev_feature in list_of_bev_features]
    image_features_shape_set = [image_feature.shape for image_feature in tuple_of_image_features]
    
    continuous_fusion = ContinuousFusion(BEV_features_shape_set, image_features_shape_set).cuda()
    for batch_ndx, sample in enumerate(data_loader):
        filtered_points_raw = sample["pointcloud_raw"].cuda()
        filtered_uv = sample["projected_loc_uv"].cuda()
        num_points = sample["num_points_raw"].cuda()
        
        list_of_bev_features = continuous_fusion(list_of_bev_features, tuple_of_image_features, filtered_points_raw, filtered_uv, num_points)
        for bev_feature in list_of_bev_features:
            print(bev_feature.shape)

# # NOT USED # #
# def get_extrinsic_parameter():
#     # translation is 0, 0, 0
#     trans = np.zeros((3,1))
#     v_lidar = np.array([  -1.57079633,    3.12042851,   -1.57079633 ])
#     v_cam = np.array([  -3.13498819,    1.59196951,    1.56942932 ])
#     v_diff = v_cam - v_lidar
#     q = quaternion.from_euler_angles(v_diff)
#     R_ = quaternion.as_rotation_matrix(q)
#     RT = np.concatenate((R_,trans), axis=-1)
#     return RT

# def get_intrinsic_parameter():
#     cameraMatrix = np.array([[268.51188197672957, 0.0, 320.0],
#                              [0.0, 268.51188197672957, 240.0], 
#                              [0.0, 0.0, 1.0]])
#     return cameraMatrix

# def bilinear_resampling(feature_map, target_uv, downscale=1):
#     """
#     feature_map: image feature map (C, H/downscale, W/downscale)
#     target_uv: location of raw image, not feature map (N,2)
#     downscale: ratio of image vs feature_map


#     """
#     u = target_uv[:,0].view(-1)/downscale
#     v = target_uv[:,1].view(-1)/downscale

#     u_lower = u.type(torch.long)
#     v_lower = v.type(torch.long)
#     u_upper = u_lower + 1
#     v_upper = v_lower + 1
#     du = u - u_lower.type(torch.float)
#     dv = v - v_lower.type(torch.float)
#     pixels_out = feature_map[:,v_lower,u_lower]*(1 - dv)*(1 - du) + \
#                  feature_map[:,v_upper,u_lower]*dv*(1 - du) + \
#                  feature_map[:,v_lower,u_upper]*(1 - dv)*du + \
#                  feature_map[:,v_upper,u_upper]*dv*du
#     return pixels_out
# # step 1: define arbitrary points
# points_x = 10*torch.rand(10000,1).cuda() # input
# points_y = torch.rand(10000,1).cuda()# input
# points_z = torch.ones(10000,1).cuda()  # input
# points_xy = torch.cat((points_x, points_y), dim=-1)
# points_xyz = torch.cat((points_xy, points_z), dim=-1)


# # step 2: project 3d lidar point to image point
# ones = torch.ones((points_xyz.shape[0],1)).cuda()
# xyz_one = torch.cat((points_xyz, ones), dim=-1) # input
# print("xyz_one.shape: ", xyz_one.shape)

# RT = get_extrinsic_parameter()
# C = get_intrinsic_parameter()
# CRT = np.matmul(C, RT)
# CRT_tensor = torch.tensor(CRT).permute(1,0).cuda().type(torch.float)
# uv_z = torch.matmul(xyz_one,CRT_tensor).permute(1,0)
# uv = uv_z/uv_z[-1]
# uv = uv[:2]
# uv = torch.where(uv[0] > 0, uv, torch.tensor(0).type(torch.float).cuda())
# uv = torch.where(uv[0] < 640, uv, torch.tensor(0).type(torch.float).cuda())
# uv = torch.where(uv[1] > 0, uv, torch.tensor(0).type(torch.float).cuda())
# uv = torch.where(uv[1] < 480, uv, torch.tensor(0).type(torch.float).cuda())
# indices = torch.nonzero(uv)
# indices = indices[:int(indices.shape[0]/2),1]
# filtered_points_raw = torch.zeros(10000,3).cuda()
# filtered_points_raw[:indices.shape[0]] = points_xyz[indices]
# filtered_points_raw = filtered_points_raw.unsqueeze(0).repeat(batch,1,1)
# filtered_uv = torch.zeros(10000,2).cuda()
# filtered_uv[:indices.shape[0]] = uv.permute(1,0)[indices]
# filtered_uv = filtered_uv.unsqueeze(0).repeat(batch,1,1)
# num_points = [indices.shape[0]]*4
# print("filtered_points_raw.shape: ", filtered_points_raw.shape)
# print("filtered_uv.shape: ",filtered_uv.shape)  
# print("length: ", indices.shape[0])