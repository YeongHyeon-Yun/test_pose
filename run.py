import cv2
import numpy as np
import tflite_pose
import tflites_pose_classifier

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break
    
    points, conf = tflite_pose.detect(image)
    tflite_pose.draw_a_pose(image, points, conf)
    cat, conf = tflites_pose_classifier.classify(points)
    print(cat, conf)

    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break