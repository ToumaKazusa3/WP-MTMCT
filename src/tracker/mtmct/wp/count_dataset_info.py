from utils_for_mtmct import DataProcesser

tracklet_dir = '/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/crop_images/yolo_nms03_all_dataset_2/'
DataProcesser.count_dataset_info(tracklet_dir=tracklet_dir, cam_num=6, verbose=True, save_info=True)

