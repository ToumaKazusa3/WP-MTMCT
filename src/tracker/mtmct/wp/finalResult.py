import pickle
import os
import os.path as osp
import numpy as np
from collections import defaultdict
import glob
import json
import utils
from utils_for_mtmct import DataPacker
from pathlib import Path

root = Path('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/')
exp_name = 'yolo_nms03_all_dataset_2_ft1'
crop_image_root = root / 'crop_images/' / exp_name
old2new = DataPacker.load(crop_image_root / 'info.pkl')['old2new']
new2cluster = DataPacker.load(crop_image_root / 'new2cluster_update_with_gps.pkl')
result_root = root / 'results/' / exp_name

for camid in range(6):
    with open(result_root / ('GPSReID0'+str(camid+1)+'_refine_v2.txt'), 'r') as f:
        lines = f.readlines()
        output = []
        for line in lines:
            line_list = line.strip().split(',')
            old_label = int(line_list[1])
            new_label = old2new[camid][old_label]
            if new_label == None:
                continue
            cluster_label = int(new2cluster[new_label]) + 1
            line_list[1] = str(cluster_label)
            res = ','.join(line_list)
            res += '\n'
            output.append(res)
    with open(result_root / ('GPSReID0'+str(camid+1) + '_update_with_gps.txt'), 'w') as f:
        f.writelines(output)