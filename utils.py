import torch

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw


def showLidarImg(ptList, threshold=0.3, imgSize=700, maxRange=120):
    img = torch.zeros(1, imgSize, imgSize)

    ptList = ptList[ptList[:, -1] > threshold]
    for pt in ptList:
        locX = int(pt[0] / maxRange * imgSize/2 + imgSize/2)
        locY = int(pt[1] / maxRange * imgSize/2 + imgSize/2)
        img[0, locY, locX] = 1
    
    img = TF.to_pil_image(img)
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
    Img = output['lidarImg']

    Img = TF.to_pil_image(Img)
    draw = ImageDraw.Draw(Img)
    for label in labels:
        h, w, l, x, y, z, phi = label[1:]
        rect = getRect(x, y, w, h, phi)
        draw.polygon([tuple(p) for p in rect], fill=100)
    Img.show()


