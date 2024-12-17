import os

import cv2

from pydantic import BaseModel
from ultralytics import YOLO
from typing import List

# COCO 关键点索引参考
COCO_KEYPOINTS = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle"    # 16
]

class Metadata(BaseModel):
    total_frames: int
    width: int
    height: int
    fps: int

class Keypoint(BaseModel):
    keypoint: str
    x: float
    y: float
    conf: float

class YoloMetadata(BaseModel):
    cls: str
    origin_shape: List[int]
    xywhn: List[float]
    xyxyn: List[float]

class Result(BaseModel):
    keypoints: List[Keypoint]
    yolo_metadata: YoloMetadata

class Frame(BaseModel):
    frame_index: int
    results: List[Result]

class MillingData(BaseModel):
    metadata: Metadata
    frames: List[Frame]

def pre_check(weight, video_path):
    if not os.path.exists(weight):
        raise FileNotFoundError(f"File not found: {weight}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

class cv2_video_capture_wrapper:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError("Video cannot be opened")
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def run(self):
        pass

def iterator_result(frame, model):
    results = model.predict(frame, conf=0.5, task='pose')
    
    for person in results:
        box_center_x, box_center_y, _w, _h = person.boxes.xywh[0]
        keypoints = person.keypoints.data[0]

        m_keypoints = []

        for i, (x, y, conf) in enumerate(keypoints):
            m_keypoint = Keypoint(
                keypoint=COCO_KEYPOINTS[i],
                x=(x - box_center_x) / (_w / 2),
                y=(y - box_center_y) / (_h / 2),
                conf=conf
            )
            m_yolo_metadata = YoloMetadata(
                cls=person.names[0],
                origin_shape=person.boxes.orig_shape,
                xywhn=person.boxes.xywhn[0],
                xyxyn=person.boxes.xyxyn[0]
            )

            m_keypoints.append(m_keypoint)
        
        yield Result(keypoints=m_keypoints, yolo_metadata=m_yolo_metadata)

def mill(weight, video_path, verbose=True):
    pre_check(weight, video_path)

    # 加载模型
    model = YOLO(weight)
    if not verbose:
        model.overrides['verbose'] = False

    with cv2_video_capture_wrapper(video_path) as cap:
        m_metadata = Metadata(
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=int(cap.get(cv2.CAP_PROP_FPS))
        )

        m_frames = []       

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = [result for result in iterator_result(frame, model)]
            m_frames.append(Frame(
                frame_index=cap.get(cv2.CAP_PROP_POS_FRAMES),
                results=results
                ))
        
        data_set = MillingData(metadata=m_metadata, frames=m_frames)
    
    return data_set

if __name__ == "__main__":
    weight = "weights/yolo11x-pose.pt"
    video_path = "images/jljt.mp4"

    data_set = mill(weight, video_path)
    print(data_set.model_dump_json())
