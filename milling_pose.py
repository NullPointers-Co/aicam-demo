import os
from typing import List

# pylint: disable=no-member
import cv2
import torch

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


# format ext
image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff']

video_extensions = ['mp4', 'avi', 'mkv', 'mov', 'flv']


# pylint: disable=missing-docstring
class Metadata(BaseModel):
    total_frames: int
    width: int
    height: int
    fps: int


# pylint: disable=missing-docstring
class Keypoint(BaseModel):
    coco_idx: int
    keypoint: str
    x: float
    y: float
    conf: float


# pylint: disable=missing-docstring
class Normalization(BaseModel):
    ptc0: List[float]
    ptc1: List[float]
    ptc0_coco_idx: int
    ptc1_coco_idx: int
    root_l: float


# pylint: disable=missing-docstring
class YoloMetadata(BaseModel):
    cls: str
    origin_shape: List[int]
    xywhn: List[float]
    xyxyn: List[float]


# pylint: disable=missing-docstring
class Result(BaseModel):
    keypoints: List[Keypoint]
    normalization: Normalization
    yolo_metadata: YoloMetadata


# pylint: disable=missing-docstring
class Frame(BaseModel):
    frame_index: int
    results: List[Result]


# pylint: disable=missing-docstring
class MillingData(BaseModel):
    metadata: Metadata
    frames: List[Frame]


def get_ext(path):
    """
    Checks if the file extension is supported.
    Parameters:
    path (str): The path to the file.
    Raises:
    ValueError: If the file extension is not supported.
    """

    name, ext = os.path.basename(path).split('.')
    if ext.lower() in image_extensions:
        return "image"
    elif ext.lower() in video_extensions:
        return "video"
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


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


# pylint: disable=missing-docstring
class cv2_image_capture_wrapper:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None

    def __enter__(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise IOError("Image cannot be opened")
        return self.image

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        # box_center_x, box_center_y, _w, _h = person.boxes.xywh[0]
        keypoints = person.keypoints.data[0]

        kps = [(i, (x, y, c)) for i, (x, y, c) in enumerate(keypoints)]
        sorted_kps = sorted(kps, key=lambda x: x[1][2], reverse=True)
        (ptc0_idx, ptc0), (ptc1_idx, ptc1) = [(i, (x, y)) for i, (x, y, c)
                                              in sorted_kps[:2]]

        root_l = torch.norm(torch.Tensor(ptc1) - torch.Tensor(ptc0))

        m_keypoints = []

        for i, (x, y, conf) in enumerate(keypoints):
            # TODO try new normalization
            # Pn = (n_norm * x1, n_norm * y1)
            n_norm = torch.norm(torch.Tensor([x, y]) - torch.Tensor(ptc0))
            x0, y0 = ptc0
            x1, y1 = ptc1
            r = n_norm / root_l
            # end try new normalization

            m_keypoint = Keypoint(
                # TODO add coco index
                # 有没有必要都算一遍norm，只除root_l够吗
                coco_idx=i,
                keypoint=COCO_KEYPOINTS[i],
                x=x / x1,
                y=y / y1,
                # x=x0 + r * ((x1 - x0) / root_l),
                # y=y0 + r * ((y1 - y0) / root_l),
                conf=conf
            )

            m_keypoints.append(m_keypoint)

        m_normalization = Normalization(
            ptc0=ptc0, ptc1=ptc1,
            ptc0_coco_idx=ptc0_idx,
            ptc1_coco_idx=ptc1_idx,
            root_l=root_l
        )

        m_yolo_metadata = YoloMetadata(
            cls=person.names[0],
            origin_shape=person.boxes.orig_shape,
            xywhn=person.boxes.xywhn[0],
            xyxyn=person.boxes.xyxyn[0]
        )

        yield Result(keypoints=m_keypoints,
                     normalization=m_normalization,
                     yolo_metadata=m_yolo_metadata)


def mill_image(weight, image_path, verbose=True):
    """
    Process an image file using a YOLO model to detect objects and return the results.
    Args:
        weight (str): Path to the YOLO model weights file.
        image_path (str): Path to the image file to be processed.
        verbose (bool, optional): If True, enables verbose mode for the YOLO model. Defaults to True.
    Returns:
        MillingData: An object containing metadata about the image and the detection results.
    """

    pre_check(weight, image_path)

    # 加载模型
    model = YOLO(weight)
    if not verbose:
        model.overrides['verbose'] = False

    with cv2_image_capture_wrapper(image_path) as image:
        m_metadata = Metadata(
            total_frames=1,
            width=image.shape[1],
            height=image.shape[0],
            fps=0
        )

        m_frames = []

        results = [result for result in iterator_result(image, model)]
        m_frames.append(Frame(
            frame_index=0,
            results=results
        ))

        data_set = MillingData(metadata=m_metadata, frames=m_frames)

    return data_set


def mill_video(weight, video_path, verbose=True):
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


def mill(weight, path, verbose=True):
    """
    Process an image or video file using a YOLO model to detect objects and return the results.
    Args:
        weight (str): Path to the YOLO model weights file.
        path (str): Path to the image or video file to be processed.
        verbose (bool, optional): If True, enables verbose mode for the YOLO model. Defaults to True.
    Returns:
        MillingData: An object containing metadata about the image or video and the detection results.
    """

    ext = get_ext(path)

    if ext == "image":
        return mill_image(weight, path, verbose)
    elif ext == "video":
        return mill_video(weight, path, verbose)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    weight = "weights/yolo11n-pose.pt"
    # video_path = "images/input1.mp4"
    # video_path = "images/output1.jpeg"
    # video_path = "images/output2.jpeg"
    for path in ("images/yoga2.jpeg", "images/yoga3.jpeg"):
        video_path = path
        data_set = mill(weight, video_path)
        # print(data_set.model_dump_json())
        print(data_set.frames[0].results[0].keypoints[0])
        print(data_set.frames[0].results[0].normalization)
