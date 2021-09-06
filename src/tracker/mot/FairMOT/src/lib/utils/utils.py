from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def generate_anchors(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx = np.meshgrid(np.arange(nGh), np.arange(nGw), indexing='ij')

    mesh = np.stack([xx, yy], axis=0)  # Shape 2, nGh, nGw
    mesh = np.tile(np.expand_dims(mesh, axis=0), (nA, 1, 1, 1)) # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = np.tile(np.expand_dims(np.expand_dims(anchor_wh, -1), -1), (1, 1, nGh, nGw))  # Shape nA x 2 x nGh x nGw
    anchor_mesh = np.concatenate((mesh, anchor_offset_mesh), axis=1)  # Shape nA x 4 x nGh x nGw
    return anchor_mesh


def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = np.log(gw/pw)
    dh = np.log(gh/ph)
    return np.stack((dx, dy, dw, dh), axis=1)

import cv2
import glob
import os
from pathlib import Path
import errno
import pickle
import json

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
        # 用refine之后的数据

        result_path = result_dir / ('GPSReID0'+str(camid)+'_refine_v2.txt')
        data_list = []
        # 读入帧ID,Tracklet ID,左上角x坐标,左上角y坐标,w,h
        with open(result_path, 'r') as f:
            total_data = f.readlines()
            for data in total_data:
                data = data.strip().split(',')
                data_list.append([int(data[0]), int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])])
        data_arr = np.array(data_list)

        # 导入full img
        for img_path in img_paths:
            # 读入index张full img，实际上就是第index帧，imgpath是有序的
            # tracking结果下标从1开始，而full img图片从0开始所以要加1
            index = int(img_path.split('/')[-1].split('.')[0]) + 1
            img0 = cv2.imread(img_path)
            # 选择第index帧的tracking数据
            data_cur = data_arr[data_arr[:,0] == index]

            for data in data_cur:
                fid, tid, x1, y1, w, h = data
                save_img_path = save_img_dir / '{:02d}'.format(camid) / '{:05d}'.format(int(tid))
                check_path(save_img_path, create=True)
                # 统计当前文件夹下文件数目作为这个tracklet中的第count张图片
                count = len(glob.glob(str(save_img_path / '*.jpg')))
                # track的结果可能会超过图片范围
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
                # 所有下标都从1开始索引
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