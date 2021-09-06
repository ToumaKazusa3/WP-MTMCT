import cv2
import numpy as np
import glob
import os
from collections import defaultdict
from tracking_utils import visualization as vis
import argparse
from utils_for_mtmct import RefineDataset

# parser = argparse.ArgumentParser()
# parser.add_argument('--i', default=1, help='camid')
# results = parser.parse_args()

# i = int(results.i)
for i in range(1,7):

    path = os.path.join('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/results/all_dataset_1/', 'GPSReID0'+str(i)+'.txt')
    pathnew = os.path.join('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/results/all_dataset_1/', 'GPSReID0'+str(i)+'_refine_v2.txt')
    refineData = []
    refineDataDict = defaultdict(list)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(',')
            locx = float(data[2]) + float(data[4]) / 2
            locy = float(data[3]) + float(data[5])
            if RefineDataset.refine_v2(i, locx, locy):
                refineData.append(line)
                refineDataDict[int(data[0])].append([int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])])
    with open(pathnew, 'w') as f:
        f.writelines(refineData)

# imgpaths = glob.glob(os.path.join('/data/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/test/GPSReID0'+str(i),'img1/','*.jpg'))
# imgpaths.sort()
# fid = 0
# for imgpath in imgpaths:
#     frameData = np.array(refineDataDict[fid+1])
#     img = cv2.imread(imgpath)
#     if not os.path.exists('/data/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/outputs/win'+str(i)+'/refine/'):
#         os.mkdir('/data/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/outputs/win'+str(i)+'/refine/')
#     if len(frameData) == 0:
#         cv2.imwrite('/data/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/outputs/win'+str(i)+'/refine/{:05d}.jpg'.format(fid), img)
#         fid += 1
#         continue
#
#     online_im = vis.plot_tracking(img, frameData[:,1:], frameData[:,0], frame_id=fid, fps=15.)
#     cv2.imwrite('/data/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/outputs/win'+str(i)+'/refine/{:05d}.jpg'.format(fid - 1), online_im)
#     fid += 1