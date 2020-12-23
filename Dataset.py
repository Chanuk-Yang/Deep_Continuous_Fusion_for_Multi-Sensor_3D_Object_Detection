import torch
import os
import time
import numpy as np
import torchvision.transforms.functional as TF

from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from utils import showLidarImg, showLidarBoundingBox


class KITTIDataset(Dataset):

    def __init__(
        self, 
        vSize=[36, 700, 700], 
        kMode=True, 
        tMode=True, 
        baselinePath='./'
    ):
        """
        args:
            vSize:
                the shape of voxelized lidar point
                dtype: list [channel, hei, wid], int64
                default: [36, 700, 700]
            
            kMode:
                kitti dataset mode.
                dtype: bool
                default: True
            
            tMode:
                trainging Mode
                dttype: bool
                default: True
            
            baselinePath:
                the parent path of kitti dataset
                dtype: string
                default: './'
        """
        super(KITTIDataset, self).__init__()
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
            label has the 8dim, corresponding to cat, h, w, length, x, y, z, rotation_y.
            Finally, preprocess the label to the voxelizaed format.

        output:
            dtype: tensor.float32 // device:[cpu]
            shape: [idx, 8]
        """
        RANGE = 80
        MINZ = 0
        MAXZ = 20

        Delta = (MAXZ - MINZ) / self.vSize[0]
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
        f = self.vSize[-1] / (2*RANGE)
        R = np.array([[0, 0, f*2], [f, 0, 0], [0, 1, 0,]])
        R1 = np.array([[1, 0, 0], [0, f * 2, 0], [0, 0, f]])
        Label = np.array(Label)
        HWL_BoundingBox = Label[:, 1:4].dot(R1)
        Location_BoundingBox = np.transpose(R.dot(np.transpose(Label[:, 4:-1]))) + np.array([0, RANGE * f, 3])
        Label[:, 4:-1] = np.clip(Location_BoundingBox, -1000, 699)
        Label[:, 1:4] = HWL_BoundingBox
        Label = torch.tensor(Label).float().cpu()
        opener.close()
        return Label
    
    def showBoundingBox(self, pillowImage, Label):
        image = ImageDraw.Draw(pillowImage)
        for label in Label:
            XY = label[4:6]
            WL = label[2:4]
            angle = label[-1]
            cosTheta = torch.cos(torch.tensor(0.0))
            sinTheta = torch.sin(torch.tensor(0.0))
            x1 = - WL[1]/2
            y1 = - WL[0]/2
            x2 = WL[1]/2
            y2 = WL[0]/2

            x1_ = x1 * cosTheta - y1 * sinTheta + XY[0]
            y1_ = x1 * sinTheta + y1 * cosTheta + XY[1]

            x2_ = x2 * cosTheta - y2 * sinTheta + XY[0]
            y2_ = x2 * sinTheta + y2 * cosTheta + XY[1]

            image.rectangle((x1_, y1_, x2_, y2_))
        
        pillowImage.show()

    def voxelizingLidarPt(self, vPath, Label):
        """
        voxelize the label.

        explain:
            voxelization means that lidar point is divided by its position.
            0. calibrate velodyne to cam02 coordinate.
            1. change the coordinator of the cam02 to the coordinator of the image
            2. voxelize the tensor based on the preprocessed lidar point.
            
        input:
            vPath: the path of lidar point
        output:
            voxelizeImg:
                dtype: torch.float32, tensor
                shape: [channel, hei, wid]
        """
        # load/calbrate the lidar point
        # lidarPt = np.fromfile(vPath, dtype=np.float32).reshape((-1, 4))
        RANGE = 80
        MINZ = 0
        MAXZ = 20
        Delta = (MAXZ - MINZ) / self.vSize[0]

        lidarPt = np.fromfile(vPath, dtype=np.float32).reshape((-1, 4))
        lidarPt = np.transpose(lidarPt)
        lidarPt = np.clip(lidarPt, -RANGE+1, RANGE-1)
        R = np.load('./KITTI/calib.npy')
        f = self.vSize[-1] / (2*RANGE)
        R1 = np.array([[0, 0, f*2, 0], [f, 0, 0, 0], [0, 1, 0, 0]])
        lidarPt = np.transpose(R1.dot(R.dot(lidarPt))) + np.array([0, RANGE * f, 3])
        index = lidarPt[:, 0] > 0
        lidarPt = lidarPt[index]
        z = np.clip(lidarPt[:, 2], MINZ, MAXZ)
        lidarPt[:, 2] = z
        lidarPt = torch.tensor(lidarPt).float().cpu()
        # voxelize the tensor based on the preprocessed lidar point.
        size = self.vSize[1] * self.vSize[2]
        voxelImg = torch.zeros(self.vSize[0] * size, dtype=torch.uint8)
        bevImg = torch.zeros(size, dtype=torch.uint8)
        labelImg = torch.zeros(size, dtype=torch.uint8)
        for i in range(self.vSize[0]):
            index = (lidarPt[:, 2] < (MINZ + Delta * (i+1))) * (lidarPt[:, 2]> (MINZ + Delta * i - 1e-3))
            pt = lidarPt[index]
            floorPt = torch.floor(pt)
            indexbev = floorPt[:, 1] * self.vSize[1] + floorPt[:, 0]
            # indexbev = torch.clamp_max(indexbev, self.vSize[-1]**2-1)
            ivoxel = i * size + indexbev
            ivoxel = index.long()
            indexbev = indexbev.long()
            voxelImg[ivoxel] = 1.0
            bevImg[indexbev] = 1
        for label in Label:
            HWL = label[1:4]
            XY = label[4:-2]
            point = torch.floor(XY[1] * self.vSize[1] + XY[0]).long()
            labelImg[point] = 1

        voxelImg = voxelImg.view((self.vSize))
        bevImg = bevImg.view((1, self.vSize[1], self.vSize[1]))
        labelImg = labelImg.view((1, self.vSize[1], self.vSize[1]))
        tempImg = torch.zeros((3, self.vSize[1], self.vSize[2]))
        tempImg[0:1, :, :] = bevImg
        tempImg[1:2, : ,:] = labelImg
        k = TF.to_pil_image(tempImg.float())
        self.showBoundingBox(k, Label)

        return voxelImg

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
        x = time.time()
        voxelImg = self.voxelizingLidarPt(vPath, Label)
        print(time.time() - x)
        # # load the image
        # img = self.loadingImage(iPath)
        
        # output = {}
        # output['img'] = img
        # output['voxelImg'] = voxelImg
        # output['label'] = Label
        # output['bevImg'] = bevImg
        # showLidarBoundingBox(output)

        # return output


if __name__ == "__main__":
    path = '/Volumes/GoogleDrive/내 드라이브/Dataset'
    datatset = KITTIDataset(baselinePath=path)
    print(len(datatset))
    datatset[83]

