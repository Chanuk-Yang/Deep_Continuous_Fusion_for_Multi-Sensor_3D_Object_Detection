import pykitti
import numpy as np

date = '2011_09_26'
drive = '0001'

data = pykitti.raw('./', date, drive)

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx  
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx  
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx  
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx  

# oxts : odometry, 즉 3차원 공간에서 6 자유도.
point_velo = np.array([0, 0, 0, 1])
point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

pt_imu = np.array([0, 0, 0, 1])
pt_w = [o.T_w_imu.dot(pt_imu) for o in data.oxts]

oxts = data.oxts
print(1)