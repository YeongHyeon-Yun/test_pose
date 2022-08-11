from cv2 import transform
# import tflite_pose
import cv2
import numpy as np
import tensorflow as tf

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


model = tf.lite.Interpreter('pose_classifier.tflite')
model.allocate_tensors()
id = model.get_input_details()
od = model.get_output_details()

def classify(points):
    # points, conf = tflite_pose.detect(image)
    points = transform_scale(points).reshape(1, 34).astype(np.float32)
    model.set_tensor(id[0]['index'], points)
    model.invoke()
    prob = model.get_tensor(od[0]['index'])
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    return pred[0], conf[0]
