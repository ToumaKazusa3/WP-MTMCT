from dataset import DataBank
from utils_for_wp import file_abs_path, DataPacker
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
    exp_name_max = exp_name[:-2]

    root = Path('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/crop_images/')
    file_path = root / exp_name / 'gt_gps_dist.pkl'
    if not file_path.is_file():
        create_dist(exp_name)
    distmat = DataPacker.load(root / exp_name / 'feature_origin.pkl')['g2g_distmat']
    gt_gps_dist = DataPacker.load(root / exp_name / 'gt_gps_dist.pkl')
    update_with_gps(distmat, gt_gps_dist, k=k, delta=delta, iters=iters, root=root / exp_name)