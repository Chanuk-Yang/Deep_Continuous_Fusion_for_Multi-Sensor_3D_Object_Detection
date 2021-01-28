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
        Range=70,
        MINZ=-2.4,
        MAXZ=0.6,
        mode="train",
        want_bev_image=True,
        baselinePath='./'
    ):
        """
        Explain
            This dataset supports for KITTI Dataset.
            You can easily download these KITTI Dataset by using
            "python DownloadData.py"
        Arguments
            Range:[float], Range of LIDAR
            MINZ:[float]
            mode:[str], check train or test mode, defalut:"train"
            want_bev_image:[bool], check Bird Eye view mode, default:True
            baselinePath:[str], set the baselinePath for KITTI dataset folder, default:"./
        """
        super(KITTIDataset, self).__init__()
        self.Range = Range
        self.MINZ = MINZ
        self.MAXZ = MAXZ
        self.want_bev_image = want_bev_image

        # set the shape of voxelization such as [Channle, Height, Width]
        self.vSize = [32, 700, 700]

        # define the Path for image, lidar point and label.
        self.iPath = os.path.join(baselinePath, 'KITTI')
        self.vPath = os.path.join(baselinePath, 'KITTI')
        tMode = mode == "train"
        if tMode:
            print("Dataset mode is Training!!")
            temp = "training"
            self.lPath = os.path.join(baselinePath, 'KITTILabel', 'training', 'label_2')            
            self.lPathList = os.listdir(self.lPath)
            self.lPathList.sort()
        else:
            print("Dataset mode is Testing")
            temp = "testing"
            
        self.iPath = os.path.join(self.iPath, temp, "image_2")
        self.vPath = os.path.join(self.vPath, temp, "velodyne")

        self.iPathList = os.listdir(self.iPath)
        self.iPathList.sort()
        self.vPathList = os.listdir(self.vPath)
        self.vPathList.sort()

        # set the category for label data.
        self.category = ['Car', 'Van', 'Truck', 'Pedestrian', "Person",
                         'Cyclist', 'Tram', 'Misc']
                         
        # set the mask for filttering the KITTI label data.
        mask = [0, 8, 9, 10, 11, 12, 13, 14]
        self.mask = [False for i in range(15)]
        for m in mask:
            self.mask[m] = True

        # set the parameter for Range
        self.Range = 70
        self.MINZ = -2.4
        self.MAXZ = 0.8
        self.f = self.vSize[-1]/(2 * self.Range)
        self.Delta = (self.MAXZ - self.MINZ) / self.vSize[0]
    
    def __len__(self):
        if len(self.iPathList) == len(self.vPathList):
            return len(self.vPathList)
        else:
            RuntimeError("# of point cloud frame is not equal to # of image frame")
            return -1
    
    def parsingLabel(self, lPath):
        """
        Explain
            the description for raw label data
                idx,    name,           Description
                0      type,           "car" or "van", etc..
                1      truncated,  
                2      occluded,       indicating occlusion state. 0 = fully visible, 3 = unkown
                3      alpha           observation angle of object, raning [-pi .. pi]
               4-7     boundingbox     2D bounding box for object in the image
                                       left, top, right, bottom pixel coordinates.
               8-10    dimensions      3D object dimensions: height, width, length (in meters)
               11-13   location        3D object location x, y, z in camera coordinates
                14     rotation_y      Rotation ry around Y-axis in camera coordinates
                15     score           only for results: indciation confidence in detection, higher is better.
            we use type, dimensions, location, rotation_y
                 
        """
        mask = [0, 8, 9, 10, 11, 12, 13, 14]
        # type, height, wdith, length, x, y, z, rotation_y
        # x,y,z,rotation_y in camera02 coordinates.
        self.mask = [False for i in range(15)]
        for m in mask:
            self.mask[m] = True

        # load the label for label file.
        opener = open(lPath, "r")
        rawLabel = opener.readlines()
        Label = []
        for line in rawLabel:
            line = np.array(line.split(' '))
            line[-1] = line[-1][:-2]
            if line[0] == 'DontCare':
                continue
            line[0] = self.category.index(line[0])
            # mask
            line = line[self.mask]
            parsedLine = np.array(line, dtype=np.float32)
            Label.append(parsedLine)
        opener.close()

        # permuting from Camera02 to Image
        RANGE = self.Range
        f = self.vSize[-1] / (2*RANGE)
        Offset = np.array([0, RANGE * f, 0])
        R1 = np.array([[0, 0, f*2], [f, 0, 0], [0, 1, 0]])
        R_LWH = np.array([[f/2, 0, 0], [0, 0, f], [0, 1, 0]])

        Label = np.array(Label)
        Label = np.concatenate((Label, np.zeros((len(Label), 1))), axis=-1)
        output = np.zeros_like(Label)
        HWL_BoundingBox = Label[:, 1:4]
        LWH_BoundingBox = HWL_BoundingBox[:, ::-1]
        LWH_BoundingBox = np.transpose(R_LWH.dot(np.transpose(HWL_BoundingBox[:, ::-1])))
        Location_BoundingBox = np.transpose(R1.dot(np.transpose(Label[:, 4:-2]))) + Offset
        orientation = Label[:, -2:-1]
        objectClass = Label[:, 0:1]
        ones = np.ones((len(objectClass), 1))
        output[:, :3] = Location_BoundingBox
        output[:, 3:6] = LWH_BoundingBox
        output[:, 6:7] = orientation
        output[:, 7:8] = objectClass
        output[:, 8:9] = ones
        output = torch.tensor(output).float().cpu()
        
        return output
    
    def showBoundingBox(self, pillowImage, Label):
        image = ImageDraw.Draw(pillowImage)
        for label in Label:
            XY = label[:2]
            LW = label[3:5]
            cosTheta = torch.cos(torch.tensor(0.0))
            sinTheta = torch.sin(torch.tensor(0.0))
            x1 = - LW[1]/2
            y1 = - LW[0]/2
            x2 = LW[1]/2
            y2 = LW[0]/2

            x1_ = x1 * cosTheta - y1 * sinTheta + XY[0]
            y1_ = x1 * sinTheta + y1 * cosTheta + XY[1]

            x2_ = x2 * cosTheta - y2 * sinTheta + XY[0]
            y2_ = x2 * sinTheta + y2 * cosTheta + XY[1]

            image.rectangle((x1_, y1_, x2_, y2_), fill=100)
        
        pillowImage.show()

    def voxelizingLidarPt(self, vPath, Label, imgSize):
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
        # set the specification for Lidar Point.
        RANGE = self.Range
        MINZ = self.MINZ
        MAXZ = self.MAXZ
        C, H, W = imgSize
        Delta = (MAXZ - MINZ) / self.vSize[0]
        f = self.vSize[-1] / (2*RANGE)
 
        lidarPt = np.fromfile(vPath, dtype=np.float32).reshape((-1, 4))  # [Num, Points], [-1, 4]
        lidarPt = np.transpose(lidarPt)  # [Points, Num], [4, -1]
        lidarPt = np.clip(lidarPt, -RANGE+1, RANGE-1)

        # calibration matrix from Velodyne point to Camera02
        R = np.load('./KITTI/calib.npy') 

        # calibration scaled-matrix from Camera02  to Image.
        R1 = np.array([[0, 0, f*2, 0], [f, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        # Offset for transfering Image coord to tensor coordinator.
        Offset = np.array([0, RANGE * f, 0, 0])

        uv = R1.dot(R.dot(lidarPt))
        imagePt = np.transpose(uv)  # [num. 3]

        uv = torch.tensor(uv[:2]).float()
        uv = torch.where(uv[0] > torch.tensor(0).float(), uv, torch.tensor(0.0).float())
        uv = torch.where(uv[0, :] < W, uv, torch.tensor(0).float())
        uv = torch.where(uv[1, :] > 0, uv, torch.tensor(0).float())
        uv = torch.where(uv[1, :] < H, uv, torch.tensor(0).float())
        uv = uv.unique(dim=1)
        indices = torch.nonzero(uv)
        indices = indices[:int(indices.shape[0]/2), 1]
        filter_points_raw = imagePt[indices]
        num_point_cloud_raw = filter_points_raw.shape[0]
        lidarPt = imagePt + Offset

        index = lidarPt[:, 0] > 0
        lidarPt = lidarPt[index]
        z = np.clip(lidarPt[:, 2], MINZ, MAXZ)
        lidarPt[:, 2] = z
        lidarPt = torch.tensor(lidarPt).float().cpu()
        # voxelize the tensor based on the preprocessed lidar point.
        size = self.vSize[1] * self.vSize[2]
        voxelImg = torch.zeros(self.vSize[0] * size, dtype=torch.uint8)
        bevImg = torch.zeros(size)
        labelImg = torch.zeros(size)
        for i in range(self.vSize[0]):
            index = (lidarPt[:, 2] <= (MINZ + Delta * (i+1))) * (lidarPt[:, 2] > (MINZ + Delta * i - 1e-3))
            pt = lidarPt[index]
            floorPt = torch.floor(pt)
            indexbev = floorPt[:, 1] * self.vSize[1] + floorPt[:, 0]
            ivoxel = i * size + indexbev
            ivoxel = index.long()
            indexbev = indexbev.long()
            voxelImg[ivoxel] = 1.0
            bevImg[indexbev] = 1.0
        for label in Label:
            XY = label[4:-2]
            point = torch.floor(XY[1] * self.vSize[1] + XY[0]).long()
            labelImg[point] = 1

        voxelImg = voxelImg.view((self.vSize)).float()

        bevImg = bevImg.view((1, self.vSize[1], self.vSize[2])).float()
        
        # labelImg = labelImg.view((1, self.vSize[1], self.vSize[1]))
        # tempImg = torch.zeros((3, self.vSize[1], self.vSize[2]))
        # tempImg[0:1, :, :] = bevImg
        # tempImg[1:2, : ,:] = labelImg
        # k = TF.to_pil_image(tempImg.float())
        # self.showBoundingBox(k, Label)
        if self.want_bev_image:
            return voxelImg, lidarPt, uv, num_point_cloud_raw, bevImg
        else:
            return voxelImg, lidarPt, uv, num_point_cloud_raw

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
        C, H, W = img.shape

        return img, (C, H, W)

    def __getitem__(self, idx):
        # configure the path of image
        iPath = self.iPathList[idx]
        iPath = os.path.join(self.iPath, iPath)
        image, iSize = self.loadingImage(iPath)

        # configure the path of velodyne point
        vPath = self.vPathList[idx]
        vPath = os.path.join(self.vPath, vPath)

        # configure the path of label
        lPath = self.lPathList[idx]
        lPath = os.path.join(self.lPath, lPath)

        # parse the label.
        Label = self.parsingLabel(lPath)

        # voxelize the label.
        if self.want_bev_image:
            voxelImg, lidarPt, uv, num_point_raw, bevImg = self.voxelizingLidarPt(vPath, Label, iSize)
            return {
                'image': image,
                "bboxes": Label,
                'num_bboxes': len(Label),
                'pointcloud': voxelImg,
                "pointcloud_raw": lidarPt,
                "projected_loc_uv": uv,
                "num_points_raw": num_point_raw,
                "lidar_bev_2Dimage": bevImg
            }
        else:
            voxelImg, lidarPt, uv, num_point_raw = self.voxelizingLidarPt(vPath, Label, iSize)
            return {
                'image': image,
                "bboxes": Label,
                'num_bboxes': len(Label),
                'pointcloud': voxelImg,
                "pointcloud_raw": lidarPt,
                "projected_loc_uv": uv,
                "num_points_raw": num_point_raw
            }


if __name__ == "__main__":
    datatset = KITTIDataset(want_bev_image=True)
    dataset = datatset[111]
    x = dataset['lidar_bev_2Dimage']
    label = dataset['bboxes']
    x = TF.to_pil_image(x)
    x.show()
    datatset.showBoundingBox(x, label)