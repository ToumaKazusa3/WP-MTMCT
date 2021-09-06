from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import cv2

import sys
sys.path.append('./lib')
sys.path.append('/home/shensj/code/WP-MTMCT/tracker/mtmct/')

from tracker.multitracker import JDETracker, joint_stracks, STrack
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
from wp.utils_for_mtmct import DataProcesser, DataPacker

from tracking_utils.utils import mkdir_if_missing
from opts import opts
from pathlib import Path


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def get_id(track):
    return track.track_id

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    # 初始化tracker
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for path, img00, imgList, img0List in dataloader:
        if frame_id % 10 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # if frame_id == 20:
        #     break

        # run tracking
        timer.tic()
        # sj:blob是输入到网络中的
        blobList = []
        for img in imgList:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            blobList.append(blob)
        # sj：img0是原图，img是经过resize和归一化之后的图
        online_targets = tracker.update(blobList, img0List, img00)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img00, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(str(save_dir / '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    # 全部的tracks
    tracks = joint_stracks(tracker.lost_stracks, tracker.removed_stracks)
    tracks = joint_stracks(tracks, tracker.tracked_stracks)
    tracks.sort(key=get_id)
    feature = []
    for track in tracks:
        feature.append(track.smooth_feat)


    return frame_id, timer.average_time, timer.calls, feature


def track_main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
               save_images=False, save_videos=False, show_image=False, save_crop_images=False, count_dataset_info=True,
               generate_refine_result=False, save_feature=False):
    logger.setLevel(logging.INFO)

    result_root = data_root / '..' / 'results' / exp_name
    save_feature_path = result_root / 'feature.pkl'
    crop_root =  data_root / '..' / 'crop_images' / exp_name
    mkdir_if_missing(result_root)
    mkdir_if_missing(crop_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    feature_all = []
    for seq in seqs:
        camid = int(seq[-1])
        output_dir = (data_root / '..' / 'outputs' / exp_name/ seq) if save_images or save_videos else None
        result_filename = result_root / '{}.txt'.format(seq)
        result_filename_refine = result_root / '{}_refine_v2.txt'.format(seq)

        logger.info('start seq: {}'.format(seq))
        # sj:creat dataloader
        dataloader = datasets.LoadImages(data_root / seq / 'img1', opt.img_size)
        # sj:read sequence infomation
        meta_info = open(data_root / seq / 'seqinfo.ini').read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        # sj:evaluate?create tracklets?
        nf, ta, tc, feature = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        feature_all.extend(feature)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        # 是否要refine数据
        if generate_refine_result:
            DataProcesser.refine_result(result_filename, result_filename_refine, camid)
        else:
            result_filename_refine = result_filename
        accs.append(evaluator.eval_file(result_filename_refine))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
            output_video_path_slow = osp.join(output_video_path[:-len(output_video_path.split('/')[-1])], 'slow256.mp4')
            cmd_str = 'ffmpeg -i output.mp4 -b:v 400k -crf 25 -y -vf "scale=1920:-1" slow.mp4'
            os.system(cmd_str)
        # 是否将tracking结果截图后保存
        if save_crop_images:
            DataProcesser.crop_image(img_dir=data_root / seq / 'img1',
                                     result_dir=result_root,
                                     save_img_dir=crop_root,
                                     camid=camid)
    if save_feature:
        feature_all = np.asarray(feature_all)
        DataPacker.dump(feature_all, save_feature_path)
    # total seqs
    if count_dataset_info:
        DataProcesser.count_dataset_info(tracklet_dir=crop_root, cam_num=6, verbose=True, save_info=False)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    # metrics = ['id_global_assignment', 'track_ratios', 'idfp']
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        # data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
        data_root = '/data5/shensj/MOT_DataSet/MOT17/test/'
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP'''
                    # '''MOT17-02-SDP
                    #   MOT17-04-SDP
                    #   MOT17-05-SDP
                    #   MOT17-09-SDP
                    #   MOT17-10-SDP
                    #   MOT17-11-SDP
                    #   MOT17-13-SDP'''
        data_root = '/data5/shensj/MOT_DataSet/MOT17/train/'
    if opt.val_mot15:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = '/data5/shensj/MOT_DataSet/MOT20/images/train'
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        # data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
        data_root = '/data5/shensj/MOT_DataSet/MOT20/images/test'
    seqs_str = '''GPSReID01
                GPSReID02
                GPSReID03
                GPSReID04
                GPSReID05
                GPSReID06'''
    # seqs_str = '''GPSReID0'''+str(opt.cam_id)
    data_root = Path('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/test/')
    seqs = [seq.strip() for seq in seqs_str.split()]
    exp_name = opt.exp_name

    track_main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=exp_name,
         show_image=False,
         save_images=True,
         save_videos=False,
         save_crop_images=True,
         count_dataset_info=True,
         generate_refine_result=True,
         save_feature=True,)
