import os

# pylint: disable=no-member
import cv2
from ultralytics import YOLO


def get_crop_coordinates(weight, file_path, verbose=True):
    """
    Calculate the crop coordinates for a video file based on YOLO model detections.
    Args:
        weight (str): Path to the YOLO model weights.
        file_path (str): Path to the video file.
        verbose (bool, optional): If True, enables verbose mode for the YOLO model. Defaults to True.
    Raises:
        FileNotFoundError: If the specified video file does not exist.
    Returns:
        None: Prints the crop coordinates and the filter_complex string for FFmpeg.
    """
    
    # Load YOLO model
    model = YOLO(weight)
    if not verbose:
        model.overrides['verbose'] = False

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FILE {file_path} NOT FOUND")

    # Open video file
    cap = cv2.VideoCapture(file_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection using YOLO model
        results = model(frame)

        max_x = 0
        max_y = 0
        min_x = w
        min_y = h

        for box in results[0].boxes:
            if int(box.cls) == 0:  # Only detect person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)

        frame_processed += 1
        print(
            f"Processing {int(frame_processed / frame_count * 100)}% Frames left: {frame_count - frame_processed} / {frame_count}", end="\r")

    cap.release()

    print(f"Max X: {max_x}, Max Y: {max_y}, Min X: {min_x}, Min Y: {min_y}")

    buffer_x = w * 0.1
    buffer_y = h * 0.1

    crop_x_start = max(0, min_x - buffer_x)
    crop_x_end = min(w, max_x + buffer_x)
    crop_y_start = max(0, min_y - buffer_y)
    crop_y_end = min(h, max_y + buffer_y)

    print(
        f"X Start: {crop_x_start}, X End: {crop_x_end}, Y Start: {crop_y_start}, Y End: {crop_y_end}")

    output_w = crop_x_end - crop_x_start
    output_h = crop_y_end - crop_y_start

    expected_w = output_h / 9 * 5
    expected_h = output_w / 5 * 9

    if expected_w > output_w:
        final_w = expected_w
        final_h = output_h
    else:
        final_w = output_w
        final_h = expected_h

    print(f"Final W: {final_w}, Final H: {final_h}")

    print(
        f"-filter_complex \"crop={output_w}:{output_h}:{crop_x_start}:{crop_y_start},pad={int(final_w)}:{int(final_h)}:(ow-iw)/2:(oh-ih)/2:black\"")


if __name__ == "__main__":
    get_crop_coordinates("weights/yolo11n.pt", "images/720p15clip.mp4", verbose=False)
