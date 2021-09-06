import os
import pickle
import cv2
import glob


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

det_path = '/data3/shensj/outputs/mtmct/det_exps/mix_yolov3_d53_mstrain-608_273e_coco/result_nms03.pkl'
with open(det_path, 'rb') as f:
    detections = pickle.load(f)
img_dir = '/data3/shensj/datasets/my_files/gps/GPSReID/full_size/'
img_paths = []
for i in range(1, 7):
    img_paths.extend(glob.glob(os.path.join(img_dir, str(i), '*.jpg')))
img_paths.sort()

count = 0
img_count = 0
camid = 1
for idx in range(len(detections)):
# for idx in range(2):
    det = detections[idx]
    img_count += 1
    if len(det) == 0:

        if img_count % 4140 == 0:
            camid += 1
        continue
    img_path = img_paths[idx]
    img = cv2.imread(img_path)
    for t in det:
        x1, y1 = int(t[0]), int(t[1])
        x2, y2 = int(t[2]), int(t[3])
        if y1 <= 0:
            y1 = 0
        if x1 <= 0:
            x1 = 0
        if y2 > 3007:
            y2 = 3007
        if x2 > 4000:
            x2 = 4000
        if not refine_v2(camid, (x1+x2)/2, y2):
            continue

        img0 = img[y1:y2, x1:x2]
        img0_save_path = os.path.join(img_path[:-23], 'yolo_nms03_det_v2_1', '{:06}.jpg'.format(count))
        cv2.imwrite(img0_save_path, img0)
        count += 1
    if img_count % 4140 == 0:
        camid += 1

