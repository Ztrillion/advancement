import time
import numpy as np
import pandas as pd
import open3d as o3d 
import multiprocessing as mp

from tqdm import tqdm
from multiprocessing import Pool
from imutils.paths import list_files

## 최상위 폴더 지정
ROOT_PATH = '/home/jovyan/autonomous_221215/sample'
print(f'ROOT_PATH\t:\t{ROOT_PATH}')

## 데이터셋 폴더 지정
dataset_path = list(list_files(f'{ROOT_PATH}'))
dataset_path.sort()

# 모든 pcd 파일의 경로
all_point_path = [pcd_file for pcd_file in dataset_path if pcd_file.endswith('.pcd')]
print(f'all_point_path length\t:\t{len(all_point_path)}\n')

# 모든 데이터셋 중 가장 많은 포인트를 찾아 반환
def max_point(s, e) :
    return max([len(o3d.io.read_point_cloud(point).points) for point in tqdm(all_point_path[s:e])])
 
if __name__ == "__main__" :
    p = Pool(processes = mp.cpu_count() // 2) 
    start = time.time()
    
    range_li = list(range(0, len(all_point_path), len(all_point_path) // (mp.cpu_count() // 2)))
    range_li[-1] = len(all_point_path)
    
    result =[]   
    for i in range(len(range_li) - 1) :
        result.append( [range_li[i], range_li[i + 1]] )
    
    print('03.calculate_max_point...')
    max_point_li = p.starmap(max_point, result)
    print(f'max_point\t:\t{max(max_point_li)}\nTime\t:\t{time.time() - start}')

    p.close()
    p.join()