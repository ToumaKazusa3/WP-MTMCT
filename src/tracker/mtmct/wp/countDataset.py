import cv2
import glob
import os
import numpy as np
import argparse
from collections import defaultdict
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--camid', default=1, type=int, help='camid')
results = parser.parse_args()

camid = results.camid

trackletPaths = os.path.join('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/crop/{:02d}'.format(camid))
trackletIDs = os.listdir(trackletPaths)
trackletIDs.sort()
# 去掉mask.pkl文件
trackletIDs = trackletIDs[:-1]
trackletsNum = len(trackletIDs)
print('total tracklets number = {}'.format(trackletsNum))
countMinTkl = 0
countAvgTklNum = 0
mask = []
totalNum = 0
for trackletID in trackletIDs:
    imgPaths = glob.glob(os.path.join(trackletPaths, trackletID, '*.jpg'))
    totalNum += len(imgPaths)
    if len(imgPaths) <= 2:
        countMinTkl += 1
        print('ID = {} have {} images'.format(trackletID, len(imgPaths)))
        mask.append(0)
        continue
    countAvgTklNum += len(imgPaths)
    mask.append(1)

print('image num less 3, num = {}'.format(countMinTkl))
print('each tracklet have {:d} image'.format(int(countAvgTklNum/trackletsNum)))
print('total image num = {}'.format(totalNum))

with open(os.path.join(trackletPaths,'mask'+str(camid)+'.pkl'), 'wb') as f:
    pickle.dump(mask, f)

with open(os.path.join(trackletPaths,'mask'+str(camid)+'.pkl'), 'rb') as f:
    data = pickle.load(f)
