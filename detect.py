import os

# pylint: disable=no-member
import cv2
from ultralytics import YOLO

# load the pre-trained model
model = YOLO('weights/yolov11n.pt')


def format_output(result):
    """
    Formats and prints the detection results.
    Args:
        result: An object containing detection results. It is expected to have the following attributes:
            - boxes: A list of detected boxes, where each box has the following attributes:
                - xyxy: A list containing the coordinates [x1, y1, x2, y2] of the top-left and bottom-right corners.
                - conf: A list containing the confidence score of the detection.
                - cls: A list containing the class index of the detected object.
            - names: A list of class names corresponding to the class indices.
    Prints:
        The class name, confidence score, and bounding box coordinates for each detected object.
        If no objects are detected, prints "No objects detected."
    """

    if result.boxes:
        for box in result.boxes:
            # 获取边界框的坐标
            x1, y1, x2, y2 = box.xyxy[0]  # 左上角和右下角坐标
            confidence = box.conf[0]  # 置信度
            class_idx = box.cls[0]  # 类别索引

            # 获取类别名称
            class_name = result.names[int(class_idx)]

            # 输出检测结果
            print(
                f"Class: {class_name}, Confidence: {confidence}, Box: [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("No objects detected.")


def get_img_mod(image_path, conf=0.35):
    """
    Processes an image using a pre-trained model and returns the modified image.
    Args:
        image_path (str): The file path to the image to be processed.
        conf (float, optional): The confidence threshold for the model's predictions. Defaults to 0.35.
    Returns:
        img_mod: The modified image after processing with the model.
    """

    _img = cv2.imread(image_path)
    _img_model = model.predict(_img, conf=conf)
    img_mod = _img_model[0]
    return img_mod


def normalize_coordinates(x1, y1, x2, y2, width, height):
    """
    Normalize the coordinates of a bounding box to be relative to the given width and height.
    Args:
        x1 (float): The x-coordinate of the top-left corner of the bounding box.
        y1 (float): The y-coordinate of the top-left corner of the bounding box.
        x2 (float): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (float): The y-coordinate of the bottom-right corner of the bounding box.
        width (float): The width of the image or space in which the bounding box is defined.
        height (float): The height of the image or space in which the bounding box is defined.
    Returns:
        tuple: A tuple containing the normalized coordinates (x1', y1', x2', y2') where each coordinate is divided by the width and height respectively.
    """

    return x1 / width, y1 / height, x2 / width, y2 / height


def denormalize_coordinates(x1, y1, x2, y2, width, height):
    """
    Denormalizes the given coordinates from a normalized scale (0 to 1) to the actual dimensions.
    Args:
        x1 (float): The normalized x-coordinate of the top-left corner.
        y1 (float): The normalized y-coordinate of the top-left corner.
        x2 (float): The normalized x-coordinate of the bottom-right corner.
        y2 (float): The normalized y-coordinate of the bottom-right corner.
        width (int): The width of the actual dimension.
        height (int): The height of the actual dimension.
    Returns:
        tuple: A tuple containing the denormalized coordinates (x1, y1, x2, y2) as integers.
    """

    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


def detect_main(target_path, reference_path, confidence):
    # get the modified images using the pre-trained model
    target_mod = get_img_mod(target_path, conf=confidence)
    reference_mod = get_img_mod(reference_path, conf=confidence)

    # draw bounding boxes on the target image
    target_img_with_box = target_mod.plot()
    # reference_img_with_box = reference_mod.plot()

    # get the original shape of the images
    target_width, target_height = target_mod.orig_shape
    reference_width, reference_height = reference_mod.orig_shape

    # print the detection results
    print("Target:")
    format_output(target_mod)
    print("Reference:")
    format_output(reference_mod)

    # compare the detected objects in the target and reference images
    for target_obj in target_mod.boxes:
        for reference_obj in reference_mod.boxes:
            if target_obj.cls[0] == reference_obj.cls[0]:
                # normalize the coordinates of the bounding boxes
                target_x1, target_y1, target_x2, target_y2 = normalize_coordinates(
                    *target_obj.xyxy[0], target_width, target_height)
                reference_x1, reference_y1, reference_x2, reference_y2 = normalize_coordinates(
                    *reference_obj.xyxy[0], reference_width, reference_height)

                # print the normalized coordinates and the movement of the bounding box
                print(
                    f"Target Box: [{target_x1}, {target_y1}, {target_x2}, {target_y2}]")
                print(
                    f"Reference Box: [{reference_x1}, {reference_y1}, {reference_x2, reference_y2}]")
                print(f"x1 move: {reference_x1 - target_x1}", f"y1 move: {reference_y1 - target_y1}",
                      f"x2 move: {reference_x2 - target_x2}", f"y2 move: {reference_y2 - target_y2}")

                # denormalize the coordinates of the reference bounding box
                retangle_x1, retangle_y1, retangle_x2, retangle_y2 = denormalize_coordinates(
                    reference_x1, reference_y1, reference_x2, reference_y2, target_width, target_height)

                # draw a green rectangle around the target object
                cv2.rectangle(target_img_with_box,
                              (retangle_x1, retangle_y1), (retangle_x2, retangle_y2),
                              (0, 255, 0), 2)  # green box with line width 2

                # save the image with the green bounding box
                origin_image_name, origin_image_ext = os.path.splitext(
                    os.path.basename(target_path))
                output_image_with_box_path = f'images/d_{origin_image_name}{origin_image_ext}'
                cv2.imwrite(output_image_with_box_path, target_img_with_box)

            else:
                print(f"Match loss: {target_obj.cls[0]}")


if __name__ == "__main__":
    # define the path to the target and reference images
    target_path = 'images/target.jpeg'
    reference_path = 'images/reference.jpeg'
    confidence = 0.35

    # process the images and compare the detected objects
    detect_main(target_path, reference_path, confidence)
