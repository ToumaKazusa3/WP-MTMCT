from dataset import DataBank
from utils_for_wp import file_abs_path, DataPacker, create_mask
from eval import update_with_gps
from pathlib import Path
from create_gt_gps_dist import create_dist
import argparse

if __name__ == '__main__':
    # wp_reid_dataset = DataBank(minframes=3)

    parser = argparse.ArgumentParser(description="GPS")
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--k', default=9, type=int)
    parser.add_argument('--delta', default=74, type=int)
    parser.add_argument('--iters', default=5, type=int)
    args = parser.parse_args()

    print('MTMCT')
    exp_name = args.exp_name
    k = args.k
    delta = args.delta
    iters = args.iters

    root = Path('/data4/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/crop_images/')
    file_path = root / exp_name / 'gt_gps_dist.pkl'
    if not file_path.is_file():
        create_dist(exp_name)
    mask_path = root / exp_name / 'mask_for_timestamp.pkl'
    if not mask_path.is_file():
        create_mask(root / exp_name)
    mask_for_timestamp = DataPacker.load(mask_path)
    distmat = DataPacker.load(root / exp_name / 'feature_origin.pkl')['g2g_distmat']
    # m, n = distmat.shape
    # for i in range(m):
    #     for j in range(n):
    #         if mask_for_timestamp[i][j] == 1 and i != j:
    #             distmat[i][j] = 1e6
    gt_gps_dist = DataPacker.load(root / exp_name / 'gt_gps_dist.pkl')
    update_with_gps(distmat, gt_gps_dist, k=k, delta=delta, iters=iters, root=root / exp_name, mask=mask_for_timestamp)