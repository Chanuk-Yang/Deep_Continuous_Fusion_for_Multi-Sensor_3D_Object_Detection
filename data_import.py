import numpy as np
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

def putBoundingBox(lidar_image, reference_bboxes, config, color=1):
    lidar_image_with_bbox = lidar_image.cpu().clone().numpy()
    img = Image.fromarray((255*lidar_image_with_bbox).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    x_scale = int(config["voxel_length"] / (config["lidar_x_max"] - config["lidar_x_min"]))
    y_scale = int(config["voxel_width"] / (config["lidar_y_max"] - config["lidar_y_min"]))
    x_offset = int(-config["lidar_x_min"] * x_scale)
    y_offset = int(-config["lidar_y_min"] * y_scale)
    for bbox in reference_bboxes:
        x = int(bbox[1]*y_scale + y_offset)
        y = int(bbox[0]*x_scale)
        width = bbox[3]*y_scale    # WARNING! IT SHOULD BE SAME SCALE IN X & Y
        height = bbox[4]*x_scale   # WARNING! IT SHOULD BE SAME SCALE IN X & Y
        angle = bbox[6] - 1.57
        rect = getRect(x=x, y=y, width=width, height=height, angle=angle)
        draw.polygon([tuple(p) for p in rect], fill=color)
    lidar_image_with_bbox = np.asarray(img)
    return torch.tensor(lidar_image_with_bbox)
