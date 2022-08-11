import tensorflow as tf
import os

import cv2
import numpy as np
import tflite_pose
import json

data_train = []
data_valid = []

# labels = ['stand', 'lunge', 'bow', 'fdb', 'heart']
# label_alt = {'launge':'lunge'}

labels = ['s', 'l', 'b', 'f', 'h']


def transform_scale(points):
    points = np.array(points)  # 17, 2
    # x1 = points[:, 0].min()
    # x2 = points[:, 0].max()
    # y1 = points[:, 1].min()
    # y2 = points[:, 1].max()

    # points[:, 0] = (points[:, 0] - x1)/ (x2 - x1) * 2 - 1
    # points[:, 1] = (points[:, 1] - y1)/ (y2 - y1) * 2 - 1

    ### 히트맵 (x1,y1), (x2,y2) 좀더 간단하게 구하는 방법 스케일링도 진행해준다.
    m1 = points.min(axis = 0)
    m2 = points.max(axis = 0)

    (points - m1) / (m2 - m1) * 2 - 1

    return points


for root, dirs, flienames in os.walk('pose08'):
    if 'train' in root:
        data = data_train
    elif 'valid' in root:
        data = data_valid
    else:
        continue
    
    for filename in flienames:

        label_txt = filename[0]

        try:  # labels에 없는 데이터 거르기!
            label = labels.index(label_txt)
        except:
            print(filename)
            continue
        
        path = os.path.join(root, filename)
        image = cv2.imread(path)
        points, conf = tflite_pose.detect(image)


        points = transform_scale(points)
        points = points.reshape(-1).tolist()
        data.append({'points':points, 'label':label})


data_total = {
    'train' : data_train,
    'valid' : data_valid
}

import json
json.dump(data_total, open('pose.json', 'w'))
