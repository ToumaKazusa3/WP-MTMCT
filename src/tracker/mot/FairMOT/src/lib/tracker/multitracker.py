from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.post_process import ctdet_post_process

from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        # 特征是融合之后的，和deepsort不一样
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # 将lost track变成activate
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        # 更新特征
        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        # opts.heads
        # hm:heatmap 预测行人的中心点(x, y)
        # wh:width and height 预测行人宽和高
        # id:行人特征维度
        # reg:offset (a, b)
        # 原图上行人bounding box就是(8x+a, 8y+b)，假设下采样8倍
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        # sj:IOU阈值
        self.det_thresh = opt.conf_thres
        # 设置最大能丢失的帧数
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        # sj:最大同屏人数
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
        # 查看输出
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
        # 坐标变换矩阵
        transxyList = [[0, 0], [1000, 0], [2000, 0],
                   [0, 750], [1000, 750], [2000, 750],
                   [0, 1500], [1000, 1500], [2000, 1500]]

        dets = None
        id_feature = None
        for im_blob, transxy in zip(blobList, transxyList):
            with torch.no_grad():
                output = self.model(im_blob)[-1]
                hm = output['hm'].sigmoid_()
                wh = output['wh']
                id_featurep = output['id']
                id_featurep = F.normalize(id_featurep, dim=1)

                reg = output['reg'] if self.opt.reg_offset else None
                # 每次检测128个框
                detsp, indsp = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
                id_featurep = _tranpose_and_gather_feat(id_featurep, indsp)
                id_featurep = id_featurep.squeeze(0)
                id_featurep = id_featurep.cpu().numpy()

            # 后处理
            detsp = self.post_process(detsp, meta)
            detsp = self.merge_outputs([detsp])[1]
            # 变换到原坐标
            detsp[..., 0:2] = detsp[..., 0:2] + np.array(transxy)
            detsp[..., 2:4] = detsp[..., 2:4] + np.array(transxy)
            # 合并所有输出
            dets = np.concatenate((dets, detsp), axis=0) if isinstance(dets, np.ndarray) else detsp
            id_feature = np.concatenate((id_feature, id_featurep), axis=0) if isinstance(id_feature, np.ndarray) else id_featurep

        # 保留大于thres的bbox
        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # 非极大值抑制
        nms_indices = nms(torch.tensor(dets[:, :4], dtype=torch.float32), torch.tensor(dets[:, 4], dtype=torch.float32), 0.2).numpy()
        dets = dets[nms_indices]
        id_feature = id_feature[nms_indices]

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

        # dets为检测结果，[N,5] N为当前帧的目标数，5为左上角点x,y,w,h,conf
        # id_feature [N, 512] N为当前帧的目标数，为特征维度
        try:
            if len(dets) > 0:
                '''Detections'''
                # 组织检测结果
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
        # 通过self.is_activated是true还是false来分unconfirmed和tracked_stracks，要求一个人至少被检测两次才会被认为是一个activated track
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # 上一帧匹配上的tracks和还没有消亡的tracks共同参加这一帧的匹配
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        # 卡尔曼滤波预测阶段，得到预测位置
        STrack.multi_predict(strack_pool)
        # 计算特征相似度， detection是检测的结果，strack_pool是之前的结果
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        # ？？？？？？？？？？？？融合运动模型，怎么融合的，没看懂
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # 二分图匹配
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        # 匹配上的track
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # 如果是上一帧该track也是tracked那么只要更新特征，并且再次将该track放入到activated_starcks
                # 更新特征，以及最后一次tracked的帧（就是当前帧）
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # 如果上一帧该track是lost状态，那么重新将其置为activate态
                track.re_activate(det, self.frame_id, new_id=False)
                # 添加到refined_stracks中
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        # 没匹配上的detections和没有匹配上的上一帧状态为tracked的track，用IOU来匹配，
        # 不用上一帧状态为lost的track，是因为位置不好估计
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # 计算iou距离
        dists = matching.iou_distance(r_tracked_stracks, detections)
        # 二分图匹配
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        # 匹配上的track更新
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                # 为什么还有这个？？？？？？？？
                assert 1 == 2, "?????????"
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            # 将剩余没有匹配上的track状态设置为lost
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
            # 将该track标记为removed
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 没有匹配上的detection，初始化为一个新的track
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        # 没有匹配上的lost track，看它是否大于最大更新帧数，如果是就标记为remove
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
        # ？？？？？？？？？？？？？去重？？？？
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

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
