import torch
import os
import h5py
from torch.utils.data import Dataset
from data_import import putBoundingBox
import time
import numpy as np
import quaternion

class CarlaDataset(Dataset):
    def __init__(self, mode="train",want_bev_image=False):
        super(CarlaDataset, self).__init__()
        self.hdf5_files = self.load_dataset(mode = mode)
        self.hdf5_id_dict = self.getIdDict(self.hdf5_files)
        self.length = 0
        self.scenario_length = []
        self.scenario_name = []
        if (want_bev_image):
            self.want_bev_image = True
        else:
            self.want_bev_image = False

        for hdf5_file in self.hdf5_files:
            single_data_scenario = self.hdf5_files[hdf5_file]
            self.length += len(single_data_scenario)
            self.scenario_name.append(hdf5_file)
            self.scenario_length.append(len(single_data_scenario))

        RT = self.get_extrinsic_parameter()
        C = self.get_intrinsic_parameter()
        CRT = np.matmul(C, RT)
        self.CRT_tensor = torch.tensor(CRT).permute(1,0).cuda().type(torch.float)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_for_scenario = idx
        if idx > self.length or idx < 0:
            RuntimeError("idx is not in data file")
            return -1
        for scenario_file_index in range(len(self.scenario_length)):
            length = self.scenario_length[scenario_file_index]
            if (idx_for_scenario - length >= 0):
                idx_for_scenario = idx_for_scenario - length
            else:
                file_name = self.scenario_name[scenario_file_index]
                data = self.hdf5_files[file_name]
                id = self.hdf5_id_dict[file_name][idx_for_scenario].strip()
                object_datas, lidar_data, image_data = self.getOneStepData(data, id)
                image_data = torch.tensor(image_data).permute(2, 0, 1).type(torch.float)
                reference_bboxes, num_reference_bboxes = self.arangeLabelData(object_datas)
                voxelized_lidar, point_cloud_raw, uv, num_points_raw = self.Voxelization_Projection(lidar_data)
                if (self.want_bev_image):
                    bev_image = self.getLidarImage(lidar_data)
                    bev_image_with_bbox = putBoundingBox(bev_image, reference_bboxes)
                    return {'image': image_data,
                            'bboxes': reference_bboxes,
                            "num_bboxes": num_reference_bboxes,
                            "pointcloud": voxelized_lidar,
                            "pointcloud_raw": point_cloud_raw,
                            "projected_loc_uv": uv,
                            "num_points_raw": num_points_raw,
                            "lidar_bev_2Dimage": bev_image_with_bbox}
                else:
                    return {'image': image_data,
                            'bboxes': reference_bboxes,
                            "num_bboxes": num_reference_bboxes,
                            "pointcloud_raw":point_cloud_raw,
                            "projected_loc_uv": uv,
                            "num_points_raw": num_points_raw,
                            "pointcloud" : voxelized_lidar}


    def load_dataset(self, mode = "train"):
        if mode == "train":
            label_path = "/media/mmc-server1/Server1/chanuk/ready_for_journal/dataset/carla_object"
        elif mode == "test":
            label_path = "/media/mmc-server1/Server1/chanuk/ready_for_journal/dataset/carla_object/test"
        else:
            print ("ERROR IN MODE TYPE, PRESS [train] OR [test] !!")
            return -1
            
                
        hdf5_files = {}
        print("reading hdf5 file...")
        file_list = os.listdir(label_path)
        for file in file_list:
            if file.split('.')[-1] == 'hdf5':
                file_dir = os.path.join(label_path, file)
                try:
                    hdf5_files[file] = h5py.File(file_dir, 'r')
                    print(file)
                except:
                    print(file + ' doesnt work. we except this folder')
        print("reading hdf5 end")
        return hdf5_files

    def valid_bbox(self, object_data):
        loc_x = object_data[0]
        loc_y = object_data[1]
        if loc_x >= 0 and loc_x < 70.0 and loc_y >= -35.0 and loc_y < 35.0:
            return True
        return False

    def valid_point(self, point):
        loc_x = int(point[-3] * 10)
        loc_y = int(point[-2] * 10 + 350)
        loc_z = int(point[-1] * 10 + 24)
        if (loc_x > 0 and loc_x < 700 and loc_y > 0 and loc_y < 700 and loc_z > 0 and loc_z < 32):
            return True
        return False

    def arangeLabelData(self, object_datas):
        """
        uint8 CLASSIFICATION_UNKNOWN=0
        uint8 CLASSIFICATION_UNKNOWN_SMALL=1
        uint8 CLASSIFICATION_UNKNOWN_MEDIUM=2
        uint8 CLASSIFICATION_UNKNOWN_BIG=3
        uint8 CLASSIFICATION_PEDESTRIAN=4
        uint8 CLASSIFICATION_BIKE=5
        uint8 CLASSIFICATION_CAR=6
        uint8 CLASSIFICATION_TRUCK=7
        uint8 CLASSIFICATION_MOTORCYCLE=8
        uint8 CLASSIFICATION_OTHER_VEHICLE=9
        uint8 CLASSIFICATION_BARRIER=10
        uint8 CLASSIFICATION_SIGN=11
        """
        ref_bboxes = torch.zeros(20,9)
        reference_bboxes = []
        i = 0
        for object_data in object_datas:
            if i>50:
                break
            if not self.valid_bbox(object_data):
                continue
            object_class = object_data[9]
            rel_x = object_data[0]
            rel_y = object_data[1]
            rel_z = object_data[2]
            ori = object_data[5]  # 3 and 4 should be carefully look whether is pitch or roll
            width = object_data[6]
            length = object_data[7]
            height = object_data[8]
            ref_bboxes[i,:] = torch.tensor([rel_x, rel_y, rel_z, length, width, height, ori, object_class, 1])
            i+=1
        return ref_bboxes, i

    def getOneStepData(self, data, id):
        image_name = 'center_image_data'
        lidar_name = 'lidar_data'
        object_data_name = 'object_data'  # relative position and rotation data
        
        lidar_data = np.array(data[id][lidar_name])
        object_data = np.array(data[id][object_data_name])
        image_data = np.array(data[id][image_name])
        
        return object_data, lidar_data, image_data

    def getIdDict(self, hdf5_files):
        hdf5_id_dict = {}
        for hdf5_file in hdf5_files:
            data_list = list(hdf5_files[hdf5_file].keys())
            hdf5_id_dict[hdf5_file] = data_list
        return hdf5_id_dict

    def get_extrinsic_parameter(self):
        # translation is 0, 0, 0
        trans = np.zeros((3,1))
        v_lidar = np.array([  -1.57079633,    3.12042851,   -1.57079633 ])
        v_cam = np.array([  -3.13498819,    1.59196951,    1.56942932 ])
        v_diff = v_cam - v_lidar
        q = quaternion.from_euler_angles(v_diff)
        R_ = quaternion.as_rotation_matrix(q)
        RT = np.concatenate((R_,trans), axis=-1)
        return RT

    def get_intrinsic_parameter(self):
        cameraMatrix = np.array([[268.51188197672957, 0.0, 320.0],
                                [0.0, 268.51188197672957, 240.0], 
                                [0.0, 0.0, 1.0]])
        return cameraMatrix

    def Projection(self, point_cloud_raw):
        point_cloud_raw = torch.tensor(point_cloud_raw).cuda()
        ones = torch.ones((point_cloud_raw.shape[0],1)).cuda()
        xyz_one = torch.cat((point_cloud_raw, ones), dim=-1) # input        
        uv_z = torch.matmul(xyz_one, self.CRT_tensor).permute(1,0)
        uv = uv_z/uv_z[-1]
        uv = uv[:2]
        uv = torch.where(uv[0] > 0, uv, torch.tensor(0).type(torch.float).cuda())
        uv = torch.where(uv[0] < 640, uv, torch.tensor(0).type(torch.float).cuda())
        uv = torch.where(uv[1] > 0, uv, torch.tensor(0).type(torch.float).cuda())
        uv = torch.where(uv[1] < 480, uv, torch.tensor(0).type(torch.float).cuda())
        indices = torch.nonzero(uv)
        indices = indices[:int(indices.shape[0]/2),1]
        filtered_points_raw = point_cloud_raw[indices]

        return uv.permute(1,0)[indices], filtered_points_raw

    def Voxelization_Projection(self, lidar_data):
        # Voxelization
        lidar_voxel = torch.zeros(32, 700, 700)
        point_cloud_raw = []
        for lidar_point in lidar_data:
            if self.valid_point(lidar_point):
                loc_x = int(lidar_point[-3] * 10)
                loc_y = int(lidar_point[-2] * 10 + 350)
                loc_z = int(lidar_point[-1] * 10 + 24)
                lidar_voxel[loc_z, loc_x, loc_y] = 1
                point_cloud_raw.append([lidar_point[-3], lidar_point[-2], lidar_point[-1]])

        # Projection
        uv, filtered_points_raw = self.Projection(point_cloud_raw)
        num_point_cloud_raw = filtered_points_raw.shape[0]
        point_cloud_raw_tensor = torch.zeros(20000, 3).cuda()
        point_cloud_raw_tensor[:num_point_cloud_raw,:] = filtered_points_raw
        uv_tensor = torch.zeros(20000, 2).cuda()
        uv_tensor[:num_point_cloud_raw,:] = uv
        return lidar_voxel, point_cloud_raw_tensor, uv_tensor, num_point_cloud_raw

    def getLidarImage(self, lidar_data):
        lidar_image = torch.zeros(1, 700, 700)
        for lidar_point in lidar_data:
            if valid_point(lidar_point):
                loc_x = int(lidar_point[-3] * 10)
                loc_y = int(lidar_point[-2] * 10 + 350)
                lidar_image[0, loc_x, loc_y] = 1
        return lidar_image

if __name__ == "__main__":
    dataset = CarlaDataset(mode="test")
    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=4,
                                          shuffle=True)
    for batch_ndx, sample in enumerate(data_loader):
        print("batch_ndx is ", batch_ndx)
        print("sample keys are ", sample.keys())
        print("bbox shape is ", sample["bboxes"].shape)
        print("num bboxes is ", sample["num_bboxes"])
        print("image shape is ", sample["image"].shape)
        print("pointcloud shape is ", sample["pointcloud"].shape)


        print("pointcloud_raw shape is ", sample["pointcloud_raw"].shape)
        print("num points is ", sample["num_points_raw"])
        print("projected_loc_uv shape is ", sample["projected_loc_uv"].shape)

        print("="*50)
        if batch_ndx >10:
            break
    # print(dataset[len(dataset)-1])