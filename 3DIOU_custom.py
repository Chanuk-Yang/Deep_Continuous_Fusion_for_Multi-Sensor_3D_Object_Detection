


'''
IOU calculation is not perfection, its too simple
YOU NEED TO MODIFY CONVEX
'''
import torch

EPSILON = 1e-5

def getCornerPoint(bbox):
    x1 = bbox[4]/2 * torch.cos(bbox[-1]) - bbox[3]/2 * torch.sin(bbox[-1]) + bbox[0]
    x2 =-bbox[4]/2 * torch.cos(bbox[-1]) - bbox[3]/2 * torch.sin(bbox[-1]) + bbox[0]
    x3 =-bbox[4]/2 * torch.cos(bbox[-1]) + bbox[3]/2 * torch.sin(bbox[-1]) + bbox[0]
    x4 = bbox[4]/2 * torch.cos(bbox[-1]) + bbox[3]/2 * torch.sin(bbox[-1]) + bbox[0]
    y1 = bbox[4]/2 * torch.sin(bbox[-1]) + bbox[3]/2 * torch.cos(bbox[-1]) + bbox[1]
    y2 =-bbox[4]/2 * torch.sin(bbox[-1]) + bbox[3]/2 * torch.cos(bbox[-1]) + bbox[1]
    y3 =-bbox[4]/2 * torch.sin(bbox[-1]) - bbox[3]/2 * torch.cos(bbox[-1]) + bbox[1]
    y4 = bbox[4]/2 * torch.sin(bbox[-1]) - bbox[3]/2 * torch.cos(bbox[-1]) + bbox[1]
    return torch.tensor([[x1,y1], [x2,y2], [x3,y3], [x4,y4]]), torch.tensor([[x2,y2], [x3,y3], [x4,y4],[x1,y1]])

# def CornerInRectangle(c1,bbox):
#     for corner in c1:

def getLineEq(c1, c1_s):
    slope = (c1_s[:,1] - c1[:,1])/(c1_s[:,0] - c1[:,0])
    return slope


def get3DIOU(bbox_1, bbox_2):
    #bbox_1, bbox_2 : size(7), x,y,z,width,length,height,orientation
    c1, c1_s = getCornerPoint(bbox_1)
    c2, c2_s = getCornerPoint(bbox_2)
    slope1 = getLineEq(c1, c1_s)
    slope2 = getLineEq(c2, c2_s)
    c_in_set = []
    for i in range(4):
        x_min = c1[i][0] if c1[i][0] < c1_s[i][0] else c1_s[i][0]
        x_max = c1[i][0] if c1[i][0] > c1_s[i][0] else c1_s[i][0]
        y_min = c1[i][1] if c1[i][1] < c1_s[i][1] else c1_s[i][1]
        y_max = c1[i][1] if c1[i][1] > c1_s[i][1] else c1_s[i][1]
        for j in range(4):
            c_in = 1/(slope2[j] - slope1[i]) * torch.matmul(torch.tensor([[-1.0, 1.0],
                                                                          [-slope2[j], slope1[i]]]),
                                                            torch.tensor([[slope1[i]*c1[i][0] - c1[i][1]],
                                                                          [slope2[i]*c2[i][0] - c2[i][1]]]))
            if x_max - x_min < EPSILON:
                if c_in[1] > y_min and c_in[1] < y_max:
                    c_in_set.append(c_in)
            elif y_max - y_min < EPSILON:
                if c_in[0] > x_min and c_in[0] < x_max:
                    c_in_set.append(c_in)
            else:
                if c_in[0] > x_min and c_in[0] < x_max and c_in[1] > y_min and c_in[1] < y_max:
                    c_in_set.append(c_in)

    IOU = 1
    return IOU

