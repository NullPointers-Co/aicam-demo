import os
from typing import List

# pylint: disable=no-member
import cv2

from pydantic import BaseModel
from ultralytics import YOLO


# COCO keypoint index reference
COCO_KEYPOINTS = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
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


# pylint: disable=missing-docstring
class Metadata(BaseModel):
    total_frames: int
    width: int
    height: int
    fps: int


# pylint: disable=missing-docstring
class Keypoint(BaseModel):
    keypoint: str
    x: float
    y: float
    conf: float


# pylint: disable=missing-docstring
class YoloMetadata(BaseModel):
    cls: str
    origin_shape: List[int]
    xywhn: List[float]
    xyxyn: List[float]


# pylint: disable=missing-docstring
class Result(BaseModel):
    keypoints: List[Keypoint]
    yolo_metadata: YoloMetadata


# pylint: disable=missing-docstring
class Frame(BaseModel):
    frame_index: int
    results: List[Result]


# pylint: disable=missing-docstring
class MillingData(BaseModel):
    metadata: Metadata
    frames: List[Frame]


def pre_check(weight, video_path):
    """
    Checks if the specified weight file and video file exist.
    Parameters:
    weight (str): The path to the weight file.
    video_path (str): The path to the video file.
    Raises:
    FileNotFoundError: If the weight file or video file does not exist.
    """

    if not os.path.exists(weight):
        raise FileNotFoundError(f"File not found: {weight}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")


# pylint: disable=missing-docstring
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
    """
    Processes a frame using a given model to predict poses and yields results.
    Args:
        frame (numpy.ndarray): The input frame to be processed.
        model (object): The model used to predict poses in the frame.
    Yields:
        Result: An object containing keypoints and YOLO metadata for each detected person.
    The Result object contains:
        keypoints (list of Keypoint): A list of keypoints for each detected person.
        yolo_metadata (YoloMetadata): Metadata related to the YOLO detection.
    The Keypoint object contains:
        keypoint (str): The name of the keypoint based on COCO_KEYPOINTS.
        x (float): The normalized x-coordinate of the keypoint.
        y (float): The normalized y-coordinate of the keypoint.
        conf (float): The confidence score of the keypoint.
    The YoloMetadata object contains:
        cls (str): The class name of the detected person.
        origin_shape (tuple): The original shape of the bounding box.
        xywhn (list): Normalized bounding box coordinates in the format (x, y, w, h).
        xyxyn (list): Normalized bounding box coordinates in the format (x1, y1, x2, y2).
    """

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
    """
    Process a video file using a YOLO model to detect objects and return the results.
    Args:
        weight (str): Path to the YOLO model weights file.
        video_path (str): Path to the video file to be processed.
        verbose (bool, optional): If True, enables verbose mode for the YOLO model. Defaults to True.
    Returns:
        MillingData: An object containing metadata about the video and the detection results for each frame.
    """

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
