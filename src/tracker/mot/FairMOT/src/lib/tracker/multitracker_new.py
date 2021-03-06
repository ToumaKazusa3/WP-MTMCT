from collections import deque

import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model, create_reid_model
from models.utils import _tranpose_and_gather_feat
from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.post_process import ctdet_post_process
from .multitracker import STrack
from collections import defaultdict
from lib.utils import transforms as T

from PIL import Image
import cv2

from .basetrack import BaseTrack, TrackState

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model

def read_det_pkl_file(det_path):
    with open(det_path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_det_file(det_path):
    det = defaultdict(list)
    with open(det_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.strip().split(',')
            det[int(line_list[0])].append([float(line_list[2]),float(line_list[3]),float(line_list[4]),float(line_list[5]),float(line_list[6])])
    return det

def crop_img_v2(img0, dets, test_transformer):
    imgList = []
    for det in dets:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        if y1 <= 0:
            y1 = 0
        if x1 <= 0:
            x1 = 0
        if y2 > 3000:
            y2 = 3000
        if x2 > 4000:
            x2 = 4000
        img = img0[y1:y2, x1:x2, :]
        # cv2.imwrite('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/outputs/yolo_all_dataset_1/GPSReID01/00000_1.jpg', img)
        try:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except:
            print(det)
            print(img.size)

        img = test_transformer(img)
        img = img.unsqueeze(0)
        imgList.append(img)

    imgs = torch.cat(imgList, dim=0)

    return imgs

def crop_img(img0, dets, test_transformer):
    imgList = []
    for det in dets:
        x1, y1, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        y2 = y1 + h
        x2 = x1 + w
        if y1 <= 0:
            y1 = 0
        if x1 <= 0:
            x1 = 0
        if y2 > 3000:
            y2 = 3000
        if x2 > 4000:
            x2 = 4000
        img = img0[y1:y2, x1:x2, :]
        try:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except:
            print(det)
            print(img.size)

        img = test_transformer(img)
        img = img.unsqueeze(0)
        imgList.append(img)

    imgs = torch.cat(imgList, dim=0)

    return imgs


class JDETracker_new(object):
    def __init__(self, opt, frame_rate=30, det_path=None, reid_model_path=None, camid=1):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        # opts.heads
        # hm:heatmap ????????????????????????(x, y)
        # wh:width and height ?????????????????????
        # id:??????????????????
        # reg:offset (a, b)
        # ???????????????bounding box??????(8x+a, 8y+b)??????????????????8???
        # self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = create_reid_model('resnet50', pretrained=False, num_features=512, num_classes=0)
        self.model.cuda()
        self.model = nn.DataParallel(self.model)

        # Load from checkpoint
        print(reid_model_path)
        checkpoint = load_checkpoint(reid_model_path)
        copy_state_dict(checkpoint['state_dict'], self.model)
        start_epoch = checkpoint['epoch']
        # best_mAP = checkpoint['best_mAP']
        print("=> Checkpoint of epoch {}".format(start_epoch))

        self.model.eval()
        # Load det
        # self.det = read_det_file(det_path)
        self.det = read_det_pkl_file(det_path)[(camid-1)*4140:camid*4140]

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        self.test_transformer = T.Compose([
            T.RectScale(256, 128),
            T.ToTensor(),
            normalizer,
        ])

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        # sj:IOU??????
        self.det_thresh = opt.conf_thres
        # ??????????????????????????????
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        # sj:??????????????????
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def vis(self, imgList, name='default'):
        count = 0
        for img in imgList:
            cv2.imwrite(osp.join('/home/shensj/code/FairMOT111/src/lib/tracker/vis', name+str(count)+'.jpg'), img)
            count += 1

    def update(self, blobList, img0List, img00):
        # ????????????
        # self.vis(img0List, 'img0List')
        # self.vis([img00], 'img00')

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0List[0].shape[1]
        height = img0List[0].shape[0]
        inp_height = blobList[0].shape[2]
        inp_width = blobList[0].shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        # dets ????????????????????????
        dets = self.det[self.frame_id-1]
        dets = np.asarray(dets)
        try:
            if len(dets) == 0:
                id_feature = None
            else:
                imgs = crop_img_v2(img00, dets, self.test_transformer)
                imgs = imgs.cuda()
                id_feature = self.model(imgs)
                id_feature = id_feature.data.cpu().numpy()
        except:
            print(self.frame_id)
            print(dets)
            print()

        # ??????????????????
        # transxyList = [[0, 0], [1000, 0], [2000, 0],
        #            [0, 750], [1000, 750], [2000, 750],
        #            [0, 1500], [1000, 1500], [2000, 1500]]
        #
        # dets = None
        # id_feature = None
        # for im_blob, transxy in zip(blobList, transxyList):
        #     with torch.no_grad():
        #         output = self.model(im_blob)[-1]
        #         hm = output['hm'].sigmoid_()
        #         wh = output['wh']
        #         id_featurep = output['id']
        #         id_featurep = F.normalize(id_featurep, dim=1)
        #
        #         reg = output['reg'] if self.opt.reg_offset else None
        #         # ????????????128??????
        #         detsp, indsp = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        #         id_featurep = _tranpose_and_gather_feat(id_featurep, indsp)
        #         id_featurep = id_featurep.squeeze(0)
        #         id_featurep = id_featurep.cpu().numpy()
        #
        #     detsp = self.post_process(detsp, meta)
        #     detsp = self.merge_outputs([detsp])[1]
        #     # ??????????????????
        #     detsp[..., 0:2] = detsp[..., 0:2] + np.array(transxy)
        #     detsp[..., 2:4] = detsp[..., 2:4] + np.array(transxy)
        #     # ??????????????????
        #     dets = np.concatenate((dets, detsp), axis=0) if isinstance(dets, np.ndarray) else detsp
        #     id_feature = np.concatenate((id_feature, id_featurep), axis=0) if isinstance(id_feature, np.ndarray) else id_featurep
        #
        # # ????????????thres???bbox
        # remain_inds = dets[:, 4] > self.opt.conf_thres
        # dets = dets[remain_inds]
        # id_feature = id_feature[remain_inds]
        #
        # # ??????????????????
        # nms_indices = nms(torch.tensor(dets[:, :4], dtype=torch.float32), torch.tensor(dets[:, 4], dtype=torch.float32), 0.2).numpy()
        # dets = dets[nms_indices]
        # id_feature = id_feature[nms_indices]

        # vis
        # '''
        # print('start vis...')
        # for i in range(0, dets.shape[0]):
        #     bbox = dets[i][0:4]
        #     cv2.rectangle(img00, (bbox[0], bbox[1]),
        #                   (bbox[2], bbox[3]),
        #                   (0, 255, 0), 2)
        # cv2.imwrite('/home/shensj/code/FairMOT111/src/lib/tracker/vis/vis.jpg', img00)
        # import time
        # time.sleep(1000000000)
        # print('sleep stop')
        # id0 = id0-1
        # '''
        try:
            if len(dets) > 0:
                '''Detections'''
                # ??????????????????
                # detections = [STrack(xywhs[:4], xywhs[4], f, 30) for
                #               (xywhs, f) in zip(dets[:, :5], id_feature)]
                detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                              (tlbrs, f) in zip(dets[:, :5], id_feature)]
            else:
                detections = []
        except IndexError:
            print('IndexError!')
            print('frame_id = ', self.frame_id)
            print('dets.shape = ', dets.shape)
            print('id_feature = ', id_feature)
            print('dets = ', dets)
            print('id_feature', id_feature)
            return []


        ''' Add newly detected tracklets to tracked_stracks'''
        # ??????self.is_activated???true??????false??????unconfirmed???tracked_stracks???????????????????????????????????????????????????????????????activated track
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # ?????????????????????tracks?????????????????????tracks??????????????????????????????
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        # ????????????????????????????????????????????????
        STrack.multi_predict(strack_pool)
        # ???????????????????????? detection?????????????????????strack_pool??????????????????
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        # ????????????????????????????????????????????????????????????????????????????????????
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # ???????????????
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        # ????????????track
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # ?????????????????????track??????tracked?????????????????????????????????????????????track?????????activated_starcks
                # ?????????????????????????????????tracked???????????????????????????
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # ??????????????????track???lost?????????????????????????????????activate???
                track.re_activate(det, self.frame_id, new_id=False)
                # ?????????refined_stracks???
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        # ???????????????detections???????????????????????????????????????tracked???track??????IOU????????????
        # ????????????????????????lost???track??????????????????????????????
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # ??????iou??????
        dists = matching.iou_distance(r_tracked_stracks, detections)
        # ???????????????
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        # ????????????track??????
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                # ?????????????????????????????????????????????
                assert 1 == 2, "?????????"
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            # ???????????????????????????track???????????????lost
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            # ??????track?????????removed
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # ??????????????????detection???????????????????????????track
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        # ??????????????????lost track???????????????????????????????????????????????????????????????remove
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # ?????????????????????????????????????????????????????????
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        # print(output_stracks)

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
