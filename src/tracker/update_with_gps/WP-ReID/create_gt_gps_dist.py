'''
计算tracklet到手机GPS之间的距离
分两步：
1. 将tracklet映射到GPS
2. 对应时间戳计算距离
'''

import pickle
import os
import os.path as osp
import numpy as np
from collections import defaultdict
import glob
import json
from utils_for_wp import DataPacker

def pixel_to_loc(data_pix2loc, traj_point, method='nearset', k=4):
    '''
    :param data_pix2loc: [pixel_x, pixel_y, jd, wd] size[n, 4]
    :param traj_list: [pixel_x, pixel_y] size[2]
    :param method: 'nearest' 'linear' 'nearest_k_mean'
    :param k: num of selected point
    :return: traj_list's jd and wd
    '''
    if method == 'nearset':
        ex = data_pix2loc[:, 0] - traj_point[0]
        ey = data_pix2loc[:, 1] - traj_point[1]
        dist = ex ** 2 + ey ** 2
        index = np.argsort(dist)[0]
        jd = data_pix2loc[index, 2]
        wd = data_pix2loc[index, 3]
    elif method == 'linear':
        index = np.where(int(data_pix2loc[:, 0]) == traj_point[0] and int(data_pix2loc[:, 1]) == traj_point[1])
        jd = data_pix2loc[index, 2]
        wd = data_pix2loc[index, 3]
    elif method == 'nearest_k_mean':
        ex = data_pix2loc[:, 0] - traj_point[0]
        ey = data_pix2loc[:, 1] - traj_point[1]
        dist = ex ** 2 + ey ** 2
        indexs = np.argsort(dist)[:k]
        jd, wd = np.mean(data_pix2loc[indexs, 2:], axis=0)
    else:
        assert 'Do not have the meathod'
    return jd, wd

def trans_gps_diff_to_dist_v1(a, b):
    '''
    :param a: ndarray:[jd, wd]
    :param b: ndarray:[jd, wd]
    :return: distance
    '''
    dist = a - b
    j_dist = dist[0] * 111000 * np.cos(a[1] / 180 * np.pi)
    w_dist = dist[1] * 111000
    return np.sqrt(np.power(j_dist, 2)+np.power(w_dist, 2))

def create_dist(exp_name):
    dataDict = defaultdict(list)
    dataList = []
    counterr = 0
    for camid in range(6):
        # DataPacker.load('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/crop_images/yolo_nms03_all_dataset_2_mmt/')
        vPath = os.path.join('/data4/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/results/', exp_name,'GPSReID0'+str(camid+1)+'_refine_v2.txt')

        v2gpsPath = os.path.join('/data3/shensj/datasets/my_files/gps/GPSReID/mapping/', str(camid+1)+'.txt')
        with open(vPath, 'r') as f:
            v2gps = np.loadtxt(v2gpsPath)
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(',')
                locx = int(float(data[2]) + float(data[4]) / 2)
                locy = int(float(data[3]) + float(data[5]))
                traj_point = np.array([locx, locy])
                try:
                    jd, wd = pixel_to_loc(v2gps, traj_point)
                    # jd, wd = v2gps[locy, locx]
                except:
                    counterr += 1
                    print('locx = ', locx)
                    print('locy = ', locy)
                    print('error = ', counterr)
                    print('-'*20)
                # camid-->frameid, trackletid, jd, wd
                dataDict[camid+1].append([int(data[0]), int(data[1]), jd, wd])
        dataDict[camid+1] = np.array(dataDict[camid+1], dtype=np.float64)
        idmax = np.max(dataDict[camid+1][:, 1])
        for i in range(int(idmax+1)):
            select_tracklet = dataDict[camid+1][dataDict[camid+1][:,1] == i]
            if select_tracklet.shape[0] <= 2:
                continue

            dataList.append(select_tracklet)

    gpsPaths = glob.glob('/data3/shensj/datasets/my_files/gps/GPSReID/trajectory/*from102630.npy')
    idChangePath = '/home/shensj/code/WP-MTMCT/mytracker/mtmct/wp/id_change.json'
    with open(idChangePath, 'r', encoding='gbk') as f:
        idChange = json.load(f)
    trajectoryDict = {}

    for gpsPath in gpsPaths:
        idOld = gpsPath.split('/')[-1].split('f')[0]
        idNew = idChange[idOld]
        data = np.load(gpsPath)
        trajectoryDict[idNew] = data

    # 新旧ID映射
    gt_gps_dist = np.zeros(shape=(len(dataList), 29))
    for i in range(len(dataList)):
        tracklet = dataList[i]
        for j in range(29):
            avgdist = 0
            count = 0
            for k in range(tracklet.shape[0]):
                frameid = tracklet[k][0]
                visual = tracklet[k][2:]
                if frameid % 6 != 0:
                    continue
                select_frame = trajectoryDict[j][trajectoryDict[j][:, 0] == frameid / 6]
                if select_frame.shape[0] == 0:
                    continue
                gps = select_frame[0,1:]
                avgdist += trans_gps_diff_to_dist_v1(gps, visual)
                count += 1
            if count == 0:
                gt_gps_dist[i,j-1] = np.inf
            else:
                gt_gps_dist[i,j-1] = avgdist / count

    with open('/data4/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/crop_images/'+exp_name+'/gt_gps_dist.pkl', 'wb') as f:
        pickle.dump(gt_gps_dist, f)

if __name__ == '__main__':
    create_dist('yolo_resnet_duke_jde_gps_ap_first')