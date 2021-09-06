import cv2
import glob
import os
import numpy as np
import argparse
from collections import defaultdict
import pickle
from utils_for_mtmct import DataPacker

tracklets_info = []
tracklet_global_id = 1
root = '/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/crop_images/yolo_nms03_all_dataset_2_ft1/'

for camid in range(6):
    tracklet_paths = os.path.join(root,'{:02d}'.format(camid+1))
    tracklet_ids = os.listdir(tracklet_paths)
    tracklet_ids.sort()
    for tracklet_id in tracklet_ids:
        img_paths = glob.glob(os.path.join(tracklet_paths, tracklet_id, '*.jpg'))
        if len(img_paths) <= 2:
            continue
        img_paths.sort()
        start_frame = int(img_paths[0].split('/')[-1].split('_')[2][1:])
        end_frame = int(img_paths[-1].split('/')[-1].split('_')[2][1:])
        tracklets_info.append([camid+1, tracklet_global_id, start_frame, end_frame])
        tracklet_global_id += 1
tracklets_info = np.array(tracklets_info)
mask_for_timestamp = np.zeros((tracklet_global_id-1, tracklet_global_id-1), dtype=np.int64)
for tracklet_id in range(1, tracklet_global_id):
    tracklet_info = tracklets_info[tracklet_id-1]
    camid = tracklet_info[0]
    start_frame = tracklet_info[2]
    end_frame = tracklet_info[3]
    tracklets_with_same_camid = tracklets_info[tracklets_info[:, 0] == camid]
    for tracklet_with_same_camid in tracklets_with_same_camid:
        if tracklet_with_same_camid[2] <= end_frame and tracklet_with_same_camid[3] >= start_frame:
            mask_for_timestamp[tracklet_id-1, tracklet_with_same_camid[1]-1] = 1

DataPacker.dump(mask_for_timestamp, os.path.join(root, 'mask_for_timestamp.pkl'))
