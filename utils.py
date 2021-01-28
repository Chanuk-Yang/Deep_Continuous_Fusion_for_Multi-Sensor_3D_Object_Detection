import torch

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw


def showLidarImg(output):
    img = output['bevImg']
    img.show()


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


def showLidarBoundingBox(output):
    labels = output['label']
    Img = output['bevImg']

    Img = TF.to_pil_image(Img)
    draw = ImageDraw.Draw(Img)
    for label in labels:
        h, w, l, x, y, z, phi = label[1:]
        rect = getRect(x, y, w, h, phi)
        draw.polygon([tuple(p) for p in rect], fill=12)
    Img.show()