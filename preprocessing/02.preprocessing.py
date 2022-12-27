import json, os, time
import numpy as np
import pandas as pd
import open3d as o3d 
import multiprocessing as mp

from tqdm import tqdm
from multiprocessing import Pool
from imutils.paths import list_files

MAX_POINT = 70081

## 최상위 폴더 지정
# Training  Test  Validation

MODE = 'Training'

ROOT_PATH = f'/home/jovyan/autonomous_221215/sample/{MODE}'

if MODE in 'Training' :
    PREP_PATH = f'/home/jovyan/autonomous_221215/sample/Training/Area_1'
elif MODE in 'Validation' :
    PREP_PATH = f'/home/jovyan/autonomous_221215/sample/Test/Area_2'
elif MODE in 'Test' : 
    PREP_PATH = f'/home/jovyan/autonomous_221215/sample/Test/Area_2'
else :
    print("MODE ERROR!")
    exit()

print(f'ROOT_PATH\t:\t{ROOT_PATH}')

## 데이터셋 폴더 지정
dataset_path = list(list_files(f'{ROOT_PATH}'))
dataset_path.sort()

# 모든 pcd, json 파일의 경로
all_point_path = [pcd_file for pcd_file in dataset_path if pcd_file.endswith('.pcd')]
all_label_path = [json_file for json_file in dataset_path if json_file.endswith('.json')]

print(f'pcd_file length : {len(all_point_path)}\n')
print(f'json_file length : {len(all_label_path)}\n')

## 라벨별로 색상 지정
color_for_class = {
    'sedan'             : [255.0, 0.0, 0.0],        
    'suv'               : [0.0, 255.0, 0.0],
    'bus'               : [0.0, 0.0, 255.0],
    'truck'             : [255.0, 255.0, 0.0],
    'bicycle'           : [0.0, 255.0, 255.0],
    'pedestrian'        : [255.0, 0.0, 255.0],
    'unknown'           : [128.0, 128.0, 128.0],
    'median'            : [0.0, 0.0, 0.0],          # 값 중복을 통해 두 클래스가 none을 가리키게 사용
    'guardrail'         : [0.0, 0.0, 0.0],
    'none'              : [0.0, 0.0, 0.0],          # median, guardrail가 lb2idx에서 none으로 처리
}

# color_for_class 출력
for k, v in color_for_class.items(): print(f'{k} | {v}')
print('\n')

color2lb = {','.join(map(str, color)) : lb for lb, color in color_for_class.items()}
lb2idx    = {lb : idx for idx, lb in enumerate(color2lb.values())}

# lb2idx 출력
for k, v in lb2idx.items(): print(f'{k} | {v}')

# 전처리 시작
print('\n\n[WAIT] preprocessing start...\n\n')

def preprocessing(s, e) :
    for idx, (point_path, label_path) in tqdm(enumerate(zip(all_point_path[s:e], all_label_path[s:e]))) :
        label_buf = {}
        color, labels = [], []
        color_append, labels_apped = color.append, labels.append
        
        pcd_point = np.asarray(o3d.io.read_point_cloud(point_path).points)
        
        with open(label_path, 'r') as fr : pcd_label = json.load(fr)

        for k in pcd_label['annotations'] :
            for voxel in k['3D_points'] :
                if voxel in pcd_point :
                    label_buf[tuple(voxel)] = k['class']

        final_point = { **dict.fromkeys(list(map(tuple, pcd_point.tolist())), 'none'), **label_buf }

        point_clouds = list(map(list, list(final_point.keys())))
        label_values = list(final_point.values())
            
        for label in range(len(label_values)) : color_append(color_for_class[label_values[label]])
        
        ## 데이터가 가장 많은 포인트 클라우드보다 적은 경우 부족한 만큼 더미 데이터 추가
        if len(point_clouds) < MAX_POINT: point_clouds += [[0, 0, 0]] * (MAX_POINT - len(point_clouds))
        if len(color) < MAX_POINT: color += [color_for_class['none']] * (MAX_POINT - len(color))
        
        colors = np.asarray(color) 
        for color in colors : labels_apped(lb2idx[color2lb[','.join(map(str, color))]])
            
        os.makedirs(f'{PREP_PATH}/{ idx + s}', exist_ok = True)
        np.save(f'{PREP_PATH}/{ idx + s}/xyzrgb.npy', np.hstack((np.asarray(point_clouds), colors)))
        np.save(f'{PREP_PATH}/{ idx + s}/label.npy', np.array(labels))

if __name__ == "__main__" :
    p = Pool(processes = mp.cpu_count() // 2) 
    start = time.time()
    
    range_li = list(range(0, len(all_point_path), len(all_point_path) // (mp.cpu_count() // 2)))
    range_li[-1] = len(all_point_path)
    
    result =[]   
    for i in range(len(range_li) - 1) :
        result.append( [range_li[i], range_li[i + 1]] )
    
    preprocess_result = p.starmap(preprocessing, result)
    print('\n\n[INFO] preprocessing done')
    print(f'\nTime\t:\t{time.time() - start}')

    p.close()
    p.join()