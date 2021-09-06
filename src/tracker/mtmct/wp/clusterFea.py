'''
'''
import os
import pickle
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import numpy as np
import torch
from utils_for_mtmct import norm_data, DataPacker
from pathlib import Path

root = Path('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/crop_images/yolo_nms03_all_dataset_2/')
data = DataPacker.load(root / 'feature_mars.pkl')
# data = DataPacker.load(root / 'g2g_distmat_update_with_gps.pkl')
gt_label = DataPacker.load(root / 'info.pkl')['old2new']
mask_for_timestamp = DataPacker.load(root / 'mask_for_timestamp.pkl')
distMat = data['g2g_distmat']
# distMat = norm_data(distMat)
distMat = -distMat
m, n = distMat.shape
for i in range(m):
    for j in range(n):
        if mask_for_timestamp[i][j] == 1 and i != j:
            distMat[i][j] = -10000

ap = AffinityPropagation(damping=0.9, affinity='precomputed').fit(distMat)
cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_

n_clusters_ = len(cluster_centers_indices)

with open(root / 'new2cluster_update_without_gps.pkl', 'wb') as f:
    pickle.dump(labels, f)