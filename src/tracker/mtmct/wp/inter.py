import numpy as np
import os.path as osp
import cv2
from scipy import interpolate

if __name__ == '__main__':
    path_img = osp.join('/data3/shensj/datasets/my_files/gps/GPSReID/full_size/1/000000.jpg')
    image = cv2.imread(path_img)
    h, w, c = image.shape
    for id in range(1,7):

        path_txt = osp.join('/data3/shensj/datasets/my_files/gps/GPSReID/mapping/'+str(id)+'.txt')
        ptl = np.loadtxt(path_txt)

        results = np.zeros(shape=(h, w, 2))
        xy = ptl[:, 0:2]
        longitude = ptl[:, 2]
        latitude = ptl[:, 3]
        grid_x, grid_y = np.mgrid[0:h, 0:w]

        grid_long = interpolate.griddata(xy, longitude, (grid_x, grid_y), method='nearest')
        grid_la = interpolate.griddata(xy, latitude, (grid_x, grid_y), method='nearest')
        results[:, :, 0] = grid_long
        results[:, :, 1] = grid_la
        np.save(osp.join('/data3/shensj/datasets/my_files/gps/GPSReID/mapping/', str(id) + 'nearest' + '.npy'), results)

        grid_long = interpolate.griddata(xy, longitude, (grid_x, grid_y), method='linear', fill_value=-1)
        grid_la = interpolate.griddata(xy, latitude, (grid_x, grid_y), method='linear', fill_value=-1)
        results[:, :, 0] = grid_long
        results[:, :, 1] = grid_la
        np.save(osp.join('/data3/shensj/datasets/my_files/gps/GPSReID/mapping/', str(id) + 'linear' + '.npy'), results)

        grid_long = interpolate.griddata(xy, longitude, (grid_x, grid_y), method='cubic', fill_value=-1)
        grid_la = interpolate.griddata(xy, latitude, (grid_x, grid_y), method='cubic', fill_value=-1)
        results[:, :, 0] = grid_long
        results[:, :, 1] = grid_la
        np.save(osp.join('/data3/shensj/datasets/my_files/gps/GPSReID/mapping/', str(id) + 'cubic' + '.npy'), results)