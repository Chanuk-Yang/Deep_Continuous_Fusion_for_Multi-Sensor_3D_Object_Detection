import torch
import os

import numpy as np
import torchvision.transforms.functional as TF

from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from utils import showLidarImg, showLidarBoundingBox


class KITTIDataset(Dataset):

    def __init__(
        self, vSize=[32, 700, 700], want_bev_image=False, 
        kMode=True, tMode=True, baselinePath='./', maxRange=120):
        super(KITTIDataset, self).__init__()

        self.vSize = vSize
        self.maxRange = maxRange
        self.voxelThresh = [1/(i+1) for i in range(vSize[0])]
        self.voxelThresh = self.voxelThresh[::-1]
        
        self.iPath = os.path.join(baselinePath, 'KITTI')
        self.vPath = os.path.join(baselinePath, 'KITTI')
        if tMode:
            temp = "training"
            self.lPath = os.path.join(baselinePath, 'KITTILabel', 'training', 'label_2')
            self.lPathList = os.listdir(self.lPath)
            self.lPathList.sort()
        else:
            temp = "testing"
            
        self.iPath = os.path.join(self.iPath, temp, "image_2")
        self.vPath = os.path.join(self.vPath, temp, "velodyne")

        self.iPathList = os.listdir(self.iPath)
        self.iPathList.sort()
        self.vPathList = os.listdir(self.vPath)
        self.vPathList.sort()

        self.category = ['Car', 'Van', 'Truck', 'Pedestrian', "Person",
                         'Cyclist', 'Tram', 'Misc']
                         
        mask = [0, 8, 9, 10, 11, 12, 13, 14]
        self.mask = [False for i in range(15)]
        for m in mask:
            self.mask[m] = True
    
    def __len__(self):
        if len(self.iPathList) == len(self.vPathList):
            return len(self.vPathList)
        else:
            RuntimeError("# of point cloud frame is not equal to # of image frame")
            return -1
    
    def __getitem__(self, idx):
        iPath = self.iPathList[idx]
        iPath = os.path.join(self.iPath, iPath)

        vPath = self.vPathList[idx]
        vPath = os.path.join(self.vPath, vPath)

        lPath = self.lPathList[idx]
        lPath = os.path.join(self.lPath, lPath)
        
        opener = open(lPath, "r")
        rawLabel = opener.readlines()
        Label = []
        
        # cat, h, w, length, x, y, z, rotation_y : 7dim
        
        for line in rawLabel:
            line = np.array(line.split(' '))
            line[-1] = line[-1][:-2]
            if line[0] == 'DontCare':
                continue
            line[0] = self.category.index(line[0])
            line = line[self.mask]
            parsedLine = np.array(line, dtype=np.float32)
            Label.append(parsedLine)
        Label = np.array(Label)  # 실제 label

        # voxelization한 것에 해당하는 label로의 변환 
        Label[:, 1:4] *= 1/self.maxRange * self.vSize[-1]
        Label[:, 4:-2] *= 1/self.maxRange * self.vSize[-1]/2 + self.vSize[-1]/2
        opener.close()
        
        scan = np.fromfile(vPath, dtype=np.float32)
        scan = np.reshape(scan, (-1, 4))
        img = torch.zeros(1, self.vSize[-2], self.vSize[-1])

        # scan = scan[scan[:, -1] > 0.01]
        for pt in scan:
            locX = int(pt[0] / self.maxRange * self.vSize[-1]/2 + self.vSize[-1]/2)
            locY = int(pt[1] / self.maxRange * self.vSize[-1]/2 + self.vSize[-1]/2)
            img[0, locY, locX] = 1
        
        vImg = torch.zeros(tuple(self.vSize))

        for i, thr in enumerate(self.voxelThresh):
            temp = scan[scan[:, -1] < thr]
            for pt in temp:
                locX = int(pt[0] / self.maxRange * self.vSize[-1]/2 + self.vSize[-1]/2)
                locY = int(pt[1] / self.maxRange * self.vSize[-1]/2 + self.vSize[-1]/2)
                vImg[i, locY, locX] = 1
            scan = scan[scan[:, -1] >= thr]
        
        output = {}
        output['lidarImg'] = img
        output['voxelImg'] = vImg
        output['label'] = Label
        showLidarBoundingBox(output)

        print(1)


if __name__ == "__main__":

    datatset = KITTIDataset()
    print(len(datatset))
    datatset[1]

