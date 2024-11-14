import cv2
from ultralytics import YOLO

segment_model = YOLO('weights/yolov11n-seg.pt')

# 进行实例分割预测
segment_results = segment_model.predict(source='images/target.jpeg', task='segment')
