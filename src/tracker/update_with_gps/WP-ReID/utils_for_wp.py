import os
import errno
import pickle
import json
import numpy as np
from pathlib import Path
import glob


__all__ = ['check_path', 'DataPacker', 'np_filter']


def np_filter(arr, *arg):
    temp_arr = arr
    for i_axis, axis_i in enumerate(arg):
        map_list = []
        for i_elem, elem_i in enumerate(axis_i):
            temp_elem_arr = temp_arr[temp_arr[:, i_axis] == elem_i]
            map_list.append(temp_elem_arr)
        temp_arr = np.concatenate(map_list, axis=0)
    return temp_arr


def check_path(folder_dir, create=False):
    folder_dir = Path(folder_dir)
    if not folder_dir.exists():
        if create:
            try:
                os.makedirs(folder_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        else:
            raise IOError
    return folder_dir


def file_abs_path(arg):
    return Path(os.path.realpath(arg)).parent


class DataPacker(object):
    @staticmethod
    def dump(info, file_path):
        check_path(Path(file_path).parent, create=True)
        with open(file_path, 'wb') as f:
            pickle.dump(info, f)
        print('Store data ---> ' + str(file_path), flush=True)

    @staticmethod
    def load(file_path):
        check_path(file_path)
        with open(file_path, 'rb') as f:
            info = pickle.load(f)
            print('Load data <--- ' + str(file_path), flush=True)
            return info

    @staticmethod
    def json_dump(info, file_path):
        check_path(Path(file_path).parent, create=True)
        with open(file_path, 'w') as f:
            json.dump(info, f)
        print('Store data ---> ' + str(file_path), flush=True)

    @staticmethod
    def json_load(file_path, acq_print=True):
        check_path(file_path)
        with open(file_path, 'r') as f:
            info = json.load(f)
            if acq_print:
                print('Load data <--- ' + str(file_path), flush=True)
            return info

def create_mask(root):
    tracklets_info = []
    tracklet_global_id = 1

    for camid in range(6):
        tracklet_paths = root / '{:02d}'.format(camid+1)
        tracklet_ids = os.listdir(tracklet_paths)
        tracklet_ids.sort()
        for tracklet_id in tracklet_ids:
            img_paths = glob.glob(str(tracklet_paths / tracklet_id / '*.jpg'))
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

    DataPacker.dump(mask_for_timestamp, root / 'mask_for_timestamp.pkl')