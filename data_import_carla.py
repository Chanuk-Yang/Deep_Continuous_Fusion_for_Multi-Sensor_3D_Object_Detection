import torch
import os
import h5py
from torch.utils.data import Dataset

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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_for_scenario = idx
        if idx > self.length or idx < 0:
            RuntimeError("idx is not in data file")
            return -1
        for scenario_file_index in range(len(self.scenario_length)):
            length = self.scenario_length[scenario_file_index]
            if (idx_for_scenario - length > 0):
                idx_for_scenario = idx_for_scenario - length
            else:
                file_name = self.scenario_name[scenario_file_index]
                data = self.hdf5_files[file_name]
                id = self.hdf5_id_dict[file_name][idx_for_scenario].strip()
                object_datas, lidar_data, image_data = self.getOneStepData(data, id)
                image_data = torch.tensor(image_data).permute(2, 0, 1).unsqueeze(0).type(torch.float)
                reference_bboxes = self.arangeLabelData(object_datas)
                voxelized_lidar = self.Voxelization(lidar_data)
                if (self.want_bev_image):
                    bev_image = self.getLidarImage(lidar_data)
                    return {'image': image_data,
                            'bboxes': reference_bboxes,
                            "pointcloud": voxelized_lidar,
                            "lidar_bev_2Dimage": bev_image}
                else:
                    return {'image': image_data,
                            'bboxes': reference_bboxes,
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

    def arangeLabelData(self, object_datas):
        reference_bboxes = []
        for object_data in object_datas:
            object_class = object_data[9]
            if (object_class > 5 and object_class < 7) or object_class == 9: # if object is vehicle
                rel_x = object_data[0]
                rel_y = object_data[1]
                rel_z = object_data[2]
                ori = object_data[5]  # 3 and 4 should be carefully look whether is pitch or roll
                width = object_data[6]
                length = object_data[7]
                height = object_data[8]
                reference_bboxes.append([rel_x, rel_y, rel_z, length, width, height, ori])
        return torch.tensor(reference_bboxes)

    def getOneStepData(self, data, id):
        image_name = 'center_image_data'
        lidar_name = 'lidar_data'
        object_data_name = 'object_data'  # relative position and rotation data
        lidar_data = data[id][lidar_name].value
        object_data = data[id][object_data_name].value
        image_data = data[id][image_name].value
        return object_data, lidar_data, image_data

    def getIdDict(self, hdf5_files):
        hdf5_id_dict = {}
        for hdf5_file in hdf5_files:
            data_list = list(hdf5_files[hdf5_file].keys())
            hdf5_id_dict[hdf5_file] = data_list
        return hdf5_id_dict

    def Voxelization(self, lidar_data):
        lidar_voxel = torch.zeros(1, 32, 700, 700)
        for lidar_point in lidar_data:
            loc_x = int(lidar_point[-3] * 10)
            loc_y = int(lidar_point[-2] * 10 + 350)
            loc_z = int(lidar_point[-1] * 10 + 24)
            if (loc_x > 0 and loc_x < 700 and loc_y > 0 and loc_y < 700 and loc_z > 0 and loc_z < 32):
                lidar_voxel[0, loc_z, loc_x, loc_y] = 1
        return lidar_voxel

    def getLidarImage(self, lidar_data):
        lidar_image = torch.zeros(1, 1, 700, 700)
        for lidar_point in lidar_data:
            loc_x = int(lidar_point[-3] * 10)
            loc_y = int(lidar_point[-2] * 10 + 350)
            if (loc_x > 0 and loc_x < 700 and loc_y > 0 and loc_y < 700):
                lidar_image[0, 0, loc_x, loc_y] = 1
        return lidar_image

if __name__ == "__main__":
    dataset = CarlaDataset()
    print(len(dataset))
    print(dataset[0])
    print(dataset[len(dataset)-1])