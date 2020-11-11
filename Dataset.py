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
        self, vSize=[32, 700, 700], 
        kMode=True, tMode=True, baselinePath='./'):
        super(KITTIDataset, self).__init__()
        """
        args:
            vSize:
                the shape of voxelized lidar point

                dtype: list [channel, wid, hei], int64
            
            kMode:
                kitti dataset mode.

                dtype: bool
            
            baselinePath:
                the parent path of kitti dataset

                dtype: string
            
            maxRange:
                the max-range of lidar

                dtype : float
        """
        self.vSize = vSize
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
    
    def parsingLabel(self, lPath):
        """
        parsing label.

        explain:
            the coordinate of label is image_02 camera's own.

            the coordinate that we select is image_02 camera's coordinate.

            label has the 8dim, corresponding to cat, h, w, length, x, y, z, rotation_y.

            Finally, preprocess the label to the voxelizaed format.

        output:
            dtype: tensor.float32 // device:[cpu]
            shape: [idx, 7]
        """
        opener = open(lPath, "r")
        rawLabel = opener.readlines()
        Label = []
        for line in rawLabel:
            line = np.array(line.split(' '))
            line[-1] = line[-1][:-2]
            if line[0] == 'DontCare':
                continue
            line[0] = self.category.index(line[0])
            line = line[self.mask]
            parsedLine = np.array(line, dtype=np.float32)
            Label.append(parsedLine)
        R = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        Label = np.array(Label)
        Label = Label.dot(R)
        Label = torch.tensor(Label).float().cpu()
        opener.close()
        Label = torch.clamp(Label, -80, 80)
        Label[:, 6] += 29
        Label[:, 4:6] = Label[:, 4] / 160 * self.vSize[-1]
        return Label

    def voxelizingLidarPt(self, vPath):
        """
        voxelize the label.

        explain:
            voxelization means that lidar point is divided by its position.
            0. calibrate velodyne to cam02 coordinate.
            1. set the max arrange of position
                x : [-80, 80]
                y : [-80, 80]
                z : [-30, 3]
            2. convert label to voxelization tensor.
        input:
            vPath: the path of lidar point
        output:
            voxelizeImg:
                dtype: torch.float32, tensor
                shape: [channel, wid, hei]
        """
        # load/calbrate the lidar point
        lidarPt = np.transpose(np.fromfile(vPath, dtype=np.float32))
        R = np.transpose(np.load('./KITTI/calib.npy'))
        lidarPt = np.transpose(lidarPt.dot(R))
        lidarPt = torch.tensor(np.reshape(lidarPt, (-1, 4))).float().cpu()
        
        # shift/clamp the lidar point
        sLidarPt = torch.clamp(lidarPt, -80, 80)
        sLidarPt[:, 2] += 29

        # convert the shifted lidar point to index for voxelization.
        sLidarPt[:, :2] = sLidarPt[:, :2] / 160 * self.vSize[-1]
        sLidarPt[:, 3] = sLidarPt[:, 3] / 32 * self.vSize[0]
        sLidarPt = torch.floor(sLidarPt)
        sLidarPt = sLidarPt[:, ::-1] 
        voxelImg = torch.zeros(self.vSize).float().cpu()
        voxelImg[sLidarPt] = 1.0

        bevImg = torch.zeros((self.vSize[1:])).long().cpu()
        bevInd = sLidarPt[:, 1:]
        bevImg[bevInd] = 1
        bevImg = torch.unsqueeze(bevImg, dim=0)

        return voxelImg, bevImg

    def loadingImage(self, iPath):
        """
        load the image

        explain:
            the shape of image is determined.
            width : 1242
            height : 375

        input:
            iPath: the path of image, str
        output:
            img: tensor.float32, [3, 375, 1242]
        """

        img = Image.open(iPath)
        img = TF.to_tensor(img)

        return img

    def __getitem__(self, idx):
        # configure the path of image
        iPath = self.iPathList[idx]
        iPath = os.path.join(self.iPath, iPath)

        # configure the path of velodyne point
        vPath = self.vPathList[idx]
        vPath = os.path.join(self.vPath, vPath)

        # configure the path of label
        lPath = self.lPathList[idx]
        lPath = os.path.join(self.lPath, lPath)

        # parse the label.
        Label = self.parsingLabel(lPath)

        # voxelize the label.
        voxelImg, bevImg = self.voxelizingLidarPt(vPath)

        # load the image
        img = self.loadingImage(iPath)
        
        output = {}
        output['img'] = img
        output['voxelImg'] = voxelImg
        output['label'] = Label
        output['bevImg'] = bevImg
        showLidarBoundingBox(output)

        return output


if __name__ == "__main__":
    datatset = KITTIDataset()
    print(len(datatset))
    datatset[1]

