from video_reid_performance import compute_video_cmc_map
from update_module import get_vision_record_dist, visual_affinity_update, trajectory_distance_update, norm_data, add_mask
from eval_tools import get_signal_match_cmc
from copy import deepcopy
import numpy as np
from utils_for_wp import file_abs_path, DataPacker


class RecurrentContextPropagationModule(object):
    def __init__(self, k, delta, iters, distMat, gt_dist, traj_id, alpha=0.5, root=None, mask=None):
        self.iters = iters
        self.k = k
        self.delta = delta
        self.alpha = alpha
        self.root = root

        self.traj_id = traj_id
        self.raw_distMat = add_mask(norm_data(distMat), mask)
        self.raw_gt_dist = gt_dist
        self.gt_dist_is_inf = np.isinf(self.raw_gt_dist)


    def rcpm(self, gt_dist_new, iteration=1):
        distMat_new = visual_affinity_update(self.raw_distMat, gt_dist=gt_dist_new.copy(), T=self.delta, alpha=self.alpha)
        # cmc_reid, mAP_reid = compute_video_cmc_map(distMat_new[self.query_index], self.query_id, self.gallery_id,
        #                                     self.query_cam_id, self.gallery_cam_id)
        gt_dist_new = trajectory_distance_update(distMat_new, self.raw_gt_dist.copy(), k=self.k)
        # cmc_SM = get_signal_match_cmc(gt_dist_new[self.query_index].copy(), self.gt_dist_is_inf[self.query_index].copy(),
        #                        self.query_id.copy(), self.traj_id.copy())
        # print('Iteration {}: ReID rank-1 {:.2f} mAP {:.2f}. Signal Matching rank-1 {:.2f}'.format(iteration,
        #                                                                                           cmc_reid[0] * 100,
        #                                                                                           mAP_reid * 100,
        #                                                                                           cmc_SM[0] * 100))
        print('Iteration {}'.format(iteration))
        return distMat_new, gt_dist_new

    def __call__(self, *args, **kwargs):
        print('K={}, Delta={}, Iteration={}'.format(self.k, self.delta, self.iters))
        # cmc_reid, mAP_reid = compute_video_cmc_map(self.raw_distMat[self.query_index], self.query_id, self.gallery_id,
        #                                  self.query_cam_id, self.gallery_cam_id)
        # cmc_SM = get_signal_match_cmc(self.raw_gt_dist[self.query_index].copy(), self.gt_dist_is_inf[self.query_index].copy(),
        #                        self.query_id.copy(), self.traj_id.copy())
        #
        # print('Iteration {}: ReID rank-1 {:.2f} mAP {:.2f}. Signal Matching rank-1 {:.2f}'.format(0, cmc_reid[0] * 100,
        #                                                                                           mAP_reid * 100,
        #                                                                                           cmc_SM[0] * 100))
        last_gt_dist = self.raw_gt_dist
        for i in range(self.iters):
            last_distMat, last_gt_dist = self.rcpm(last_gt_dist, iteration=i+1)
        test = {}
        test['g2g_distmat'] = last_distMat
        DataPacker.dump(test, self.root / 'g2g_distmat_update_with_gps.pkl')
        x = 1



def update_with_gps(distMat, gt_gps_dist ,k=8, delta=74, iters=5, root=None, mask=None):
    # K是近邻，delta是判断是否用距离矩阵更新特征矩阵的阈值，iter是迭代次数，distMat是特征矩阵，gt_dist是tracklet到手机GPS的距离，traj_id是手机GPS对于的行人ID
    rcpm = RecurrentContextPropagationModule(k=k, delta=delta, iters=iters, distMat=distMat,
                                             gt_dist=gt_gps_dist, traj_id=None, root=root, mask=mask)
    rcpm()

