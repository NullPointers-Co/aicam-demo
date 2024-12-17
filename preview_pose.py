# pylint: disable=no-member
import cv2
from ultralytics import YOLO


def preview(weights, video_path, out_file_path, origin=False, box=False, verbose=True):
    """
    Preview pose estimation on a video using a pre-trained YOLO model.
    Args:
        weights (str): Path to the pre-trained YOLO model weights.
        video_path (str): Path to the input video file.
        out_file_path (str): Path to save the output video file with pose estimation.
        origin (bool, optional): If True, draw the center of the bounding box. Defaults to False.
        box (bool, optional): If True, draw the bounding box around detected persons. Defaults to False.
    Returns:
        None
    """

    model = YOLO(weights)
    if not verbose:
        model.overrides['verbose'] = False

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

    # Open video stream
    cap = cv2.VideoCapture(video_path)

    # Check if the video is opened
    if not cap.isOpened():
        print("Error: Failed to open video.")
        exit()

    # Get video information
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    out = cv2.VideoWriter(
        out_file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform pose estimation
        results = model.predict(frame, conf=0.5, task='pose')

        # Draw keypoints and bounding box
        for person in results:
            box_center_x, box_center_y, _w, _h = person.boxes.xywh[0]
            box_xyxy = person.boxes.xyxy[0]
            keypoints = person.keypoints.data[0]
            for i, (x, y, conf) in enumerate(keypoints):
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                if origin:
                    cv2.circle(frame, (int(box_center_x), int(
                        box_center_y)), 5, (0, 0, 255), -1)
                if box:
                    cv2.rectangle(frame, (int(box_xyxy[0]), int(box_xyxy[1])),
                                  (int(box_xyxy[2]), int(box_xyxy[3])),
                                  (255, 0, 0), 2)

            # Draw skeleton
            skeleton = [
                (0, 1), (1, 2), (0, 2), (1, 3), (2, 4),  # head
                (0, 5), (0, 6),  # head -> shoulder
                (5, 6), (6, 12), (12, 11), (11, 5),  # body
                (5, 7), (7, 9),  # left arm
                (6, 8), (8, 10),  # right arm
                (11, 13), (13, 15),  # left leg
                (12, 14), (14, 16)  # right leg
            ]
            for start, end in skeleton:
                if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
                    x1, y1 = keypoints[start][:2]
                    x2, y2 = keypoints[end][:2]
                    cv2.line(frame, (int(x1), int(y1)),
                             (int(x2), int(y2)), (255, 0, 0), 2)

        # Write frame to output video
        out.write(frame)

    cap.release()
    out.release()
