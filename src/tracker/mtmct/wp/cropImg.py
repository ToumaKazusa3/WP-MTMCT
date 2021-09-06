import cv2
import glob
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--camid', default=1, type=int, help='camid')
results = parser.parse_args()

camid = results.camid

imgPaths = glob.glob('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/test/GPSReID0'+str(camid)+'/img1/*.jpg')
imgPaths.sort()
trackPath = '/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/results/win'+str(camid)+'/GPSReID0'+str(camid)+'_refine.txt'
dataList = []
with open(trackPath, 'r') as f:
    totalData = f.readlines()
    for data in totalData:
        data = data.strip().split(',')
        dataList.append([int(data[0]), int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])])
dataArr = np.array(dataList)

for imgPath in imgPaths:
    index = int(imgPath.split('/')[-1].split('.')[0]) + 1
    img0 = cv2.imread(imgPath)
    dataCur = dataArr[dataArr[:,0] == index]

    for data in dataCur:
        frameid, trackletid, x1, y1, w, h = data
        savePath = os.path.join('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/crop_all_dataset_1/{:02d}'.format(camid),
                                '{:05d}'.format(int(trackletid)))
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        count = len(glob.glob(os.path.join(savePath, '*.jpg')))
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
        imgName = 'T{:05d}_C{:02d}_F{:06d}_I{:06d}.jpg'.format(int(trackletid), camid, int(frameid), count+1)
        try:
            cv2.imwrite(os.path.join(savePath, imgName), img)
        except:
            print(os.path.join(savePath, imgName))
            print(img.shape)
            print(y1)
            print(h)