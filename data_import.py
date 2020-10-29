import numpy as np
import random
import torch
from PIL import Image, ImageDraw


def arangeLabelData(object_datas):
    reference_bboxes = []
    for object_data in object_datas:
        object_class = object_data[9]
        # if (object_class > 5 and object_class < 7) or object_class == 9: # if object is vehicle
        rel_x = object_data[0]
        rel_y = object_data[1]
        rel_z = object_data[2]
        ori = object_data[5] # 3 and 4 should be carefully look whether is pitch or roll
        width = object_data[6]
        length = object_data[7]
        height = object_data[8]
        reference_bboxes.append([rel_x, rel_y, rel_z, length, width, height, ori])
    return reference_bboxes

def getOneStepData(data, id):
    image_name = 'center_image_data'
    lidar_name = 'lidar_data'
    object_data_name = 'object_data' # relative position and rotation data
    lidar_data = data[id][lidar_name].value
    object_data = data[id][object_data_name].value
    image_data = data[id][image_name].value
    return object_data, lidar_data, image_data

def getIdDict (hdf5_files):
    hdf_id_dict_train = {}
    hdf_id_dict_test = {}
    test_files = ['_2020-06-23-18-49-56.bag.hdf5', '_2020-06-24-14-52-05.bag.hdf5']
    for hdf5_file in hdf5_files:
        data_list = list(hdf5_files[hdf5_file].keys())
        random.shuffle(data_list)
        if hdf5_file in test_files:
            hdf_id_dict_test[hdf5_file] = data_list
        else:
            hdf_id_dict_train[hdf5_file] = data_list
    return hdf_id_dict_train, hdf_id_dict_test

def getRect(x, y, width, height, angle):
    rect = np.array([(-width/2, -height/2), (width/2, -height/2),
                    (width/2, height/2), (-width/2, height/2),
                     (-width/2, -height/2)])
    theta = angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect

def putBoundingBox(lidar_image, object_data):
    lidar_image = 0.5*lidar_image.cpu().clone().numpy()
    lidar_image_with_bbox = np.tile(lidar_image, (3, 1, 1))
    reference_bboxes = arangeLabelData(object_data)
    img = Image.fromarray(lidar_image_with_bbox[0])
    draw = ImageDraw.Draw(img)
    for bbox in reference_bboxes:
        x = int(bbox[1]*10+350)
        y = int(bbox[0]*10)
        width = bbox[-4]*10
        height = bbox[-3]*10
        angle = bbox[-1] - 1.57
        rect = getRect(x=x, y=y, width=width, height=height, angle=angle)
        draw.polygon([tuple(p) for p in rect], fill=1)
    lidar_image_with_bbox[0] = np.asarray(img)
    return torch.tensor(lidar_image_with_bbox)

def getLidarImage(lidar_data):
    lidar_image = torch.zeros(1, 1, 700, 700)
    for lidar_point in lidar_data:
        loc_x = int(lidar_point[-3]*10)
        loc_y = int(lidar_point[-2]*10 + 350)
        if (loc_x > 0 and loc_x < 700 and loc_y > 0 and loc_y < 700):
            lidar_image[0, 0, loc_x, loc_y] = 1
    return lidar_image

def Voxelization(lidar_data):
    lidar_voxel = torch.zeros(1, 32, 700, 700)
    for lidar_point in lidar_data:
        loc_x = int(lidar_point[-3] * 10)
        loc_y = int(lidar_point[-2] * 10 + 350)
        loc_z = int(lidar_point[-1] * 10 + 24)
        if (loc_x > 0 and loc_x < 700 and loc_y > 0 and loc_y < 700 and loc_z > 0 and loc_z < 32):
            lidar_voxel[0, loc_z, loc_x, loc_y] = 1
    return lidar_voxel
