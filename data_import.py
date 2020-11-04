import numpy as np
import random
import torch
from PIL import Image, ImageDraw

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

def putBoundingBox(lidar_image, reference_bboxes):
    lidar_image = 0.5*lidar_image.cpu().clone().numpy()
    lidar_image_with_bbox = np.tile(lidar_image, (3, 1, 1))
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
