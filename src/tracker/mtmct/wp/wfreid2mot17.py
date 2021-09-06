import glob
from functools import reduce
from utils_for_mtmct import DataPacker, DataProcesser

id = 6

name2id = DataPacker.json_load('id_change.json', encoding='gbk')

label_paths = glob.glob('/data3/shensj/datasets/my_files/gps/GPSReID/labels/'+str(id)+'/*.json')
label_paths.sort()
new_label = []
for label_path in label_paths:
    data = DataPacker.json_load(label_path, encoding='UTF-8', acq_print=False)
    objs = data['outputs']['object']
    frameid = int(data['path'].split('\\')[-1].split('.')[0]) + 1
    for obj in objs:
        name, bndbox = obj['name'], obj['bndbox']
        pid = name2id[name]
        if DataProcesser.refine_v2(id, locx=(bndbox['xmax']+bndbox['xmin'])/2, locy=bndbox['ymax']) == False:
            elem = [frameid, pid, bndbox['xmin'], bndbox['ymin'],
                    bndbox['xmax'] - bndbox['xmin'], bndbox['ymax'] - bndbox['ymin'],
                    0, 13, 1]
        else:
            elem = [frameid, pid, bndbox['xmin'], bndbox['ymin'],
                    bndbox['xmax'] - bndbox['xmin'], bndbox['ymax'] - bndbox['ymin'],
                    1, 1, 1]
        elem = reduce(lambda x,y:str(x)+str(',')+str(y), elem)
        elem += '\n'
        new_label.append(elem)

with open('/data3/shensj/datasets/my_files/gps/GPSReID/MOT/GPSReID/test/GPSReID0'+str(id)+'/gt/refine_gt_v2.txt', 'w', encoding='UTF-8') as f:
    f.writelines(new_label)