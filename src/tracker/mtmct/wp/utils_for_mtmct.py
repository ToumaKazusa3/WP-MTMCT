import json
import pickle
import numpy as np
import os
import errno
from geopy.distance import geodesic
from pathlib import Path
import glob
import cv2

def ham_to_dem(time):
    '''
    :param time: hour minute second
    :return: second
    '''
    dam_time_list = []
    if isinstance(time, list):
        for i in time:
            dem_time = int(i % 100) + 60 * (int(i % 10000) - int(i % 100)) / 100 + int(i / 10000) * 60 * 60
            dam_time_list.append(dem_time)
        return dam_time_list
    else:
        dem_time = int(time % 100) + 60 * (int(time % 10000) - int(time % 100)) / 100 + int(time / 10000) * 60 * 60
    return int(dem_time)

def get_time(time):
    '''
    :param time: hour minute second
    :return: second from 102630
    '''
    now_time = ham_to_dem(time)
    start_time = ham_to_dem(102630)
    sub_time = now_time - start_time
    return  int(sub_time)

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

def trans_gps_diff_to_dist_v2(dist):
    '''
    :param dist: [jd_sub, wd_sub]
    :return: distance
    '''
    j_dist = dist[0] * 111000 * np.cos(31 / 180 * np.pi)
    w_dist = dist[1] * 111000
    return np.sqrt(np.power(j_dist, 2)+np.power(w_dist, 2))

def trans_gps_diff_to_dist_v3(gps_1, gps_2):
    '''
    :param gps_1: [jd, wd]
    :param gps_2: [jd, wd]
    :return: distance between two gps
    '''
    return geodesic((gps_1[1], gps_1[0]), (gps_2[1], gps_2[0]).m)

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

def norm_data(a):
    '''
    :param a: feature distance N X N
    :return: normalize feature
    '''
    a = a.copy()
    a_min = np.min(a, axis=1, keepdims=True)
    _range = np.max(a,axis=1,keepdims=True) - a_min
    return (a - a_min) / _range

def check_path(folder_dir, create=False):
    '''
    :param folder_dir: file path
    :param create: create file or not
    :return:
    '''
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

class DataPacker(object):
    '''
    this class supplys four different Data processing format
    '''
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
    def json_dump(info, file_path, encoding='UTF-8'):
        check_path(Path(file_path).parent, create=True)
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(info, f)
        print('Store data ---> ' + str(file_path), flush=True)

    @staticmethod
    def json_load(file_path, encoding='UTF-8', acq_print=True):
        check_path(file_path)
        with open(file_path, 'r', encoding=encoding) as f:
            info = json.load(f)
            if acq_print:
                print('Load data <--- ' + str(file_path), flush=True)
            return info

    @staticmethod
    def np_save_txt(info, file_path):
        check_path(Path(file_path).parent, create=True)
        assert file_path.split('.')[-1] == 'txt', 'This file\' suffix is not txt, please check file path.'
        np.savetxt(file_path, info)
        print('Store data ---> ' + str(file_path), flush=True)

    @staticmethod
    def np_load_txt(file_path, acq_print=True):
        check_path(file_path)
        assert file_path.split('.')[-1] == 'txt', 'This file\' suffix is not txt, please check file path.'
        info = np.loadtxt(file_path)
        if acq_print:
            print('Load data <--- ' + str(file_path), flush=True)
        return info

    @staticmethod
    def np_save(info, file_path):
        check_path(Path(file_path).parent, create=True)
        assert file_path.split('.')[-1] == 'npy', 'This file\' suffix is not npy, please check file path.'
        np.save(file_path, info)
        print('Store data ---> ' + str(file_path), flush=True)

    @staticmethod
    def np_load(file_path, acq_print=True):
        check_path(file_path)
        assert file_path.split('.')[-1] == 'npy', 'This file\' suffix is not npy, please check file path.'
        info = np.load(file_path)
        if acq_print:
            print('Load data <--- ' + str(file_path), flush=True)
        return info

class DataProcesser(object):
    @staticmethod
    def refine_v1(camid, locx, locy):
        if camid == 1:
            if locx + 30 * locy <= 3000 or locx - 8.235 * locy >= 1200:
                return False
        if camid == 2:
            if locy <= 150 or locx >= 3750:
                return False
        if camid == 3:
            if locy <= 200:
                return False
        if camid == 4:
            if locy <= 600:
                return False
        if camid == 5:
            if locy <= 150 or 0.1167 * locx + locy <= 350 or locx - 5.7143 * locy >= 2000:
                return False
        if camid == 6:
            if locy <= 150 or locx - 2.5 * locy >= 2000:
                return False
        return True

    @staticmethod
    def refine_v2(camid, locx, locy):
        if camid == 1:
            if locx + 20 * locy <= 4000 or locx - 8.235 * locy >= 1200:
                return False
        if camid == 2:
            if locy <= 200 or locx >= 3600:
                return False
        if camid == 3:
            if locy <= 250:
                return False
        if camid == 4:
            if locy <= 600 or (locx >= 2000 and 0.5833 * locx + locy <= 2340):
                return False
        if camid == 5:
            if locy <= 150 or 0.1167 * locx + locy <= 350 or locx - 5.7143 * locy >= 2000:
                return False
        if camid == 6:
            if locy <= 150 or locx - 2.5 * locy >= 2000:
                return False
        return True

    @staticmethod
    def refine_result(path, path_new, camid):
        refineData = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(',')
                locx = float(data[2]) + float(data[4]) / 2
                locy = float(data[3]) + float(data[5])
                if DataProcesser.refine_v2(camid, locx, locy):
                    refineData.append(line)
        with open(path_new, 'w') as f:
            f.writelines(refineData)

    @staticmethod
    def crop_image(img_dir, result_dir, save_img_dir, camid):

        img_paths = glob.glob(str(img_dir / '*.jpg'))
        img_paths.sort()

        result_path = result_dir / ('GPSReID0'+str(camid)+'_refine_v2.txt')
        data_list = []
        with open(result_path, 'r') as f:
            total_data = f.readlines()
            for data in total_data:
                data = data.strip().split(',')
                data_list.append([int(data[0]), int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])])
        data_arr = np.array(data_list)

        for img_path in img_paths:
            index = int(img_path.split('/')[-1].split('.')[0]) + 1
            img0 = cv2.imread(img_path)
            data_cur = data_arr[data_arr[:,0] == index]

            for data in data_cur:
                fid, tid, x1, y1, w, h = data
                save_img_path = save_img_dir / '{:02d}'.format(camid) / '{:05d}'.format(int(tid))
                check_path(save_img_path, create=True)
                count = len(glob.glob(str(save_img_path / '*.jpg')))
                y2 = y1 + h
                x2 = x1 + w
                if y1 <= 0:
                    y1 = 0
                if x1 <= 0:
                    x1 = 0
                if y2 > 3007:
                    y2 = 3007
                if x2 > 4000:
                    x2 = 4000
                img = img0[int(y1):int(y2),int(x1):int(x2)]
                img_name = 'T{:05d}_C{:02d}_F{:06d}_I{:06d}.jpg'.format(int(tid), camid, int(fid), count+1)
                try:
                    cv2.imwrite(str(save_img_path / img_name), img)
                except:
                    print('ignore image -- ', save_img_path / img_name)

    @staticmethod
    def count_dataset_info(tracklet_dir, cam_num, verbose=False, save_info=False):
        tracklet_dir = Path(tracklet_dir)
        info = {'old2new':[{} for i in range(cam_num)], 'ignore':[],
                'tracklets_info':[{} for i in range(cam_num)], 'tracklets':[]}

        new_id = 0
        for camid in range(cam_num):
            tracklets_per_cam_dir = tracklet_dir / '{:02d}'.format(camid + 1)
            tracklet_ids = os.listdir(tracklets_per_cam_dir)
            tracklet_ids.sort()
            tracklets_num = len(tracklet_ids)

            count_ignore_tkl = 0
            count_ava_tkl = 0
            count_avg_tkl_num = 0
            total_num = 0

            for tracklet_id in tracklet_ids:
                img_paths = glob.glob(str(tracklets_per_cam_dir / tracklet_id / '*.jpg'))
                img_paths.sort()
                if len(img_paths) <= 2:
                    count_ignore_tkl += 1
                    info['old2new'][camid][int(tracklet_id)] = None
                    info['ignore'].append(tracklets_per_cam_dir / tracklet_id)
                else:
                    count_ava_tkl += 1
                    info['old2new'][camid][int(tracklet_id)] = new_id
                    new_id += 1
                    info['tracklets'].append((img_paths, new_id, camid))
                    total_num += len(img_paths)
                    count_avg_tkl_num += 1

            info['tracklets_info'][camid]['total_tracklets_num'] = tracklets_num
            info['tracklets_info'][camid]['ignore_tracklets_num'] = count_ignore_tkl
            info['tracklets_info'][camid]['available_tracklets_num'] = count_ava_tkl
            info['tracklets_info'][camid]['available_images_num'] = total_num
            info['tracklets_info'][camid]['average_images_num'] = total_num // count_ava_tkl

            assert info['tracklets_info'][camid]['total_tracklets_num'] == \
                   info['tracklets_info'][camid]['ignore_tracklets_num'] + \
                   info['tracklets_info'][camid]['available_tracklets_num']

        if save_info:
            DataPacker.dump(info, tracklet_dir / 'info.pkl')
        return info

        if verbose:
            ti = info['tracklets_info']
            to_tn = ti[1]['total_tracklets_num'] + ti[2]['total_tracklets_num'] + ti[3]['total_tracklets_num'] + \
                    ti[4]['total_tracklets_num'] + ti[5]['total_tracklets_num'] + ti[0]['total_tracklets_num']
            to_itn = ti[1]['ignore_tracklets_num'] + ti[2]['ignore_tracklets_num'] + ti[3]['ignore_tracklets_num'] + \
                     ti[4]['ignore_tracklets_num'] + ti[5]['ignore_tracklets_num'] + ti[0]['ignore_tracklets_num']
            to_atn = ti[1]['available_tracklets_num'] + ti[2]['available_tracklets_num'] + ti[3]['available_tracklets_num'] + \
                     ti[4]['available_tracklets_num'] + ti[5]['available_tracklets_num'] + ti[0]['available_tracklets_num']
            to_avain = ti[1]['available_images_num'] + ti[2]['available_images_num'] + ti[3]['available_images_num'] + \
                       ti[4]['available_images_num'] + ti[5]['available_images_num'] + ti[0]['available_images_num']
            to_avein = ti[1]['average_images_num'] + ti[2]['average_images_num'] + ti[3]['average_images_num'] + \
                       ti[4]['average_images_num'] + ti[5]['average_images_num'] + ti[0]['average_images_num']
            print("=> GPSMOT loaded")
            print("Dataset statistics:")
            print("  ----------------------------------------------------------------------------------------------------------------")
            print("  subset   | # tkls num | # ignore tkls num | # available tkls num | # available images num | # average images num")
            print("  ----------------------------------------------------------------------------------------------------------------")
            print("  cam1     | {:10d} | {:17d} | {:20d} | {:22d} | {:20d} ".format(ti[0]['total_tracklets_num'], ti[0]['ignore_tracklets_num'], \
                                                                              ti[0]['available_tracklets_num'], ti[0]['available_images_num'], ti[0]['average_images_num']))
            print("  cam2     | {:10d} | {:17d} | {:20d} | {:22d} | {:20d} ".format(ti[1]['total_tracklets_num'], ti[1]['ignore_tracklets_num'], \
                                                                              ti[1]['available_tracklets_num'], ti[1]['available_images_num'], ti[1]['average_images_num']))
            print("  cam3     | {:10d} | {:17d} | {:20d} | {:22d} | {:20d} ".format(ti[2]['total_tracklets_num'], ti[2]['ignore_tracklets_num'], \
                                                                              ti[2]['available_tracklets_num'], ti[2]['available_images_num'], ti[2]['average_images_num']))
            print("  cam4     | {:10d} | {:17d} | {:20d} | {:22d} | {:20d} ".format(ti[3]['total_tracklets_num'], ti[3]['ignore_tracklets_num'], \
                                                                              ti[3]['available_tracklets_num'], ti[3]['available_images_num'], ti[3]['average_images_num']))
            print("  cam5     | {:10d} | {:17d} | {:20d} | {:22d} | {:20d} ".format(ti[4]['total_tracklets_num'], ti[4]['ignore_tracklets_num'], \
                                                                              ti[4]['available_tracklets_num'], ti[4]['available_images_num'], ti[4]['average_images_num']))
            print("  cam6     | {:10d} | {:17d} | {:20d} | {:22d} | {:20d} ".format(ti[5]['total_tracklets_num'], ti[5]['ignore_tracklets_num'], \
                                                                              ti[5]['available_tracklets_num'], ti[5]['available_images_num'], ti[5]['average_images_num']))
            print("  ----------------------------------------------------------------------------------------------------------------")
            print("  total    | {:10d} | {:17d} | {:20d} | {:22d} | {:20d} ".format(to_tn, to_itn, to_atn, to_avain, to_avein))