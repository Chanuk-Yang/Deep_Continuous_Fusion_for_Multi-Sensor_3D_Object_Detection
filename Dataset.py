import torch
import os

import numpy as np
import torchvision.transforms.functional as TF

from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class KITTIDataset(Dataset):

    def __init__(
        self, iSize=416, vSize=[32, 700, 700], 
        kMode=True, tMode=True, baselinePath='./'):

        super(KITTIDataset, self).__init__()
        
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
                         
        mask = [0, 8, 9, 10, 11, 12, 13, -1]
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
        opener.close()
        
        opener = open(vPath, 'rb')
        x = opener.read(1)
        rawCPt = opener.readlines()
        cPt = []

        for line in rawCPt:
            line = np.array(line, dtype=np.float32)
            cPt.append(line)
        opener.close()


if __name__ == "__main__":

    datatset = KITTIDataset()
    print(len(datatset))
    datatset[1]

