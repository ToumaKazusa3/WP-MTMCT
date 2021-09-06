import os
import numpy as np
import glob
from utils_for_mtmct import norm_data, DataPacker
from pathlib import Path
from collections import defaultdict

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

    return mask_for_timestamp

def find_cluster_pos(clusters, t):
    for i in range(len(clusters)):
        if t in clusters[i]:
            return i
    return -1

def is_time_conflict(mask_for_timestamp, trajectory_i, trajectory_j):
    for i in trajectory_i:
        for j in trajectory_j:
            if mask_for_timestamp[i, j] == 1:
                return True
    return False

def cluster_tracklets(root, save_name, fea_name, crop_image_name='crop_images', exp_name='demo', thresh=0.3):
    root = Path(root)
    root = root / crop_image_name / exp_name
    mask_for_timestamp = create_mask(root)
    data = DataPacker.load(root / fea_name)
    distMat = data['g2g_distmat']
    n = distMat.shape[0]
    distMat_update = (distMat + distMat.T) / 2
    vector_update = distMat_update.reshape(-1)
    sorted_idxs = np.argsort(vector_update)
    clusters = [[i] for i in range(n)]
    for idx in sorted_idxs:
        if vector_update[idx] > thresh:
            break
        i, j = idx % n, idx // n
        # same tracklet
        if i == j:
            continue
        # same cluster
        i_pos, j_pos = find_cluster_pos(clusters, i), find_cluster_pos(clusters, j)
        assert i_pos != -1
        assert j_pos != -1
        if i_pos == j_pos:
            continue
        trajectory_i = clusters[i_pos]
        trajectory_j = clusters[j_pos]
        # time conflict
        if is_time_conflict(mask_for_timestamp, trajectory_i, trajectory_j):
            continue
        clusters.remove(trajectory_i)
        clusters.remove(trajectory_j)
        trajectory_i.extend(trajectory_j)
        clusters.append(trajectory_i)
    labels = np.zeros(n, dtype=np.int)
    for i in range(n):
        pos = find_cluster_pos(clusters, i)
        labels[i] = pos
    DataPacker.dump(labels, root / save_name)

def cosine_similarity(a, b):
    b_norm =b / np.linalg.norm(b,keepdims=True)
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm)


def select_samples(root, new2cluster_name, fea_origin_name, info='info.pkl', crop_image_name='crop_images', exp_name='demo', theta=0.80):
    crop_image_root = root / crop_image_name / exp_name
    old2new = DataPacker.load(crop_image_root / info)['old2new']
    new2cluster = DataPacker.load(crop_image_root / new2cluster_name)
    feature = DataPacker.load(crop_image_root / fea_origin_name)['gf']
    feature_single = DataPacker.load(crop_image_root / fea_origin_name)['gf_s']
    new_dataset_tmp = defaultdict(list)
    count = 0
    for cid in range(6):
        for old_tid in old2new[cid].keys():
            new_tid = old2new[cid][old_tid]
            if new_tid == None:
                continue
            cluster_tid = new2cluster[new_tid]
            new_dataset_tmp[cluster_tid].append([crop_image_root / '{:02d}'.format(cid+1) / '{:05d}'.format(old_tid),
                                            feature[count], new_tid])
            count += 1
    assert count == feature.shape[0]
    assert feature.shape[0] == len(feature_single)
    # paths, pid, camid
    new_dataset = []
    count = 0
    for k in new_dataset_tmp.keys():
        num = len(new_dataset_tmp[k])
        feature = 0
        for i in range(num):
            feature += new_dataset_tmp[k][i][1]
        feature /= num
        paths = []
        for i in range(num):
            img_dir = new_dataset_tmp[k][i][0]
            img_path = glob.glob(os.path.join(str(img_dir), '*.jpg'))
            img_path.sort()
            gf_s = feature_single[new_dataset_tmp[k][i][2]]
            c_s = cosine_similarity(gf_s, feature)
            mask = c_s > theta
            for i in range(mask.shape[0]):
                if mask[i] == 0:
                    continue
                paths.append(img_path[i])
                count+=1
        new_dataset.append([paths, k, 0])
    count_id = 0
    count_ignore = 0
    remove_idx = []
    for i in range(len(new_dataset)):
        if len(new_dataset[i][0]) == 0:
            count_ignore += 1
            remove_idx.append(i)
            continue
        count_id += 1
    for idx in remove_idx:
        new_dataset.pop(idx)
    for i in range(len(new_dataset)):
        new_dataset[i][1] = i
    new_datatset_dict = {}
    new_datatset_dict['new_dataset'] = new_dataset
    new_datatset_dict['ids_num'] = count_id

    DataPacker.dump(new_datatset_dict, crop_image_root / 'new_dataset_dict.pkl')