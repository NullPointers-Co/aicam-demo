import os
import cv2
from ultralytics import YOLO

# 初始化 YOLO 姿态模型
model = YOLO('weights/yolov11n-pose.pt')

# COCO 关键点索引参考
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 定义过滤关键点的置信度阈值
keypoint_conf_threshold = 0.5

# 过滤低置信度关键点，保持 `torch.Tensor` 数据格式
def filter_keypoints(keypoints_tensor):
    # 获取置信度列
    conf = keypoints_tensor[:, 2]
    # 创建过滤掩码：置信度大于等于阈值，且坐标不为(0, 0)
    mask = (conf >= keypoint_conf_threshold) & ((keypoints_tensor[:, 0] != 0) | (keypoints_tensor[:, 1] != 0))
    # 应用掩码过滤关键点，保留符合条件的点
    return keypoints_tensor[mask]

def format_output(result):
    if result.boxes:
        for box, keypoints in zip(result.boxes, result.keypoints.data):
            # 获取边界框的坐标
            x1, y1, x2, y2 = box.xyxy[0]  # 左上角和右下角坐标
            confidence = box.conf[0]  # 置信度
            class_idx = box.cls[0]  # 类别索引
            
            # 获取类别名称
            class_name = result.names[int(class_idx)]
            
            # 打印检测结果
            print(f"Class: {class_name}, Confidence: {confidence}, Box: [{x1}, {y1}, {x2}, {y2}]")

            # 打印关键点信息，按照COCO定义顺序，过滤低置信度点
            filtered_keypoints = filter_keypoints(keypoints)
            print("Keypoints:")
            for i, (x, y, kp_conf) in enumerate(filtered_keypoints):
                print(f"  Keypoint {i} ({COCO_KEYPOINTS[i] if i < len(COCO_KEYPOINTS) else 'unknown'}): ({x}, {y}), Confidence: {kp_conf}")

def get_img_mod(image_path, conf=0.5):
    _img = cv2.imread(image_path)
    _img_model = model.predict(_img, conf=conf, task='pose')
    img_mod = _img_model[0]
    return img_mod

# 将关键点坐标归一化为比例坐标
def normalize_coordinates(x, y, width, height):
    return x / width, y / height

# 将归一化坐标反归一化为目标图像的实际坐标
def denormalize_coordinates(x, y, width, height):
    return int(x * width), int(y * height)

def advise_pose(target_mod, reference_mod, target_img_with_box, target_width, target_height, reference_width, reference_height):
    # 归一化后过滤关键点
    target_keypoints = [
        normalize_coordinates(x, y, target_width, target_height) + (conf,)
        for (x, y, conf) in filter_keypoints(target_mod.keypoints.data[0])
    ]
    reference_keypoints = [
        normalize_coordinates(x, y, reference_width, reference_height) + (conf,)
        for (x, y, conf) in filter_keypoints(reference_mod.keypoints.data[0])
    ]
    
    for i, (target_x, target_y, target_conf) in enumerate(target_keypoints):
        if i >= len(reference_keypoints):
            print(f"Skipping Keypoint {i} due to reference keypoints shortage.")
            continue

        reference_x, reference_y, reference_conf = reference_keypoints[i]

        # 计算归一化坐标下的移动建议
        dx = reference_x - target_x
        dy = reference_y - target_y
        print(f"Keypoint {i} ({COCO_KEYPOINTS[i] if i < len(COCO_KEYPOINTS) else 'unknown'}): Move x: {dx}, Move y: {dy}")

        # 将建议坐标反归一化以绘制箭头
        target_x_denorm, target_y_denorm = denormalize_coordinates(target_x, target_y, target_width, target_height)
        reference_x_denorm, reference_y_denorm = denormalize_coordinates(reference_x, reference_y, target_width, target_height)

        # 在目标图像上绘制箭头以指示建议
        cv2.arrowedLine(
            target_img_with_box,
            (target_x_denorm, target_y_denorm),
            (reference_x_denorm, reference_y_denorm),
            (255, 0, 0), 2, tipLength=0.3
        )

# 主执行块
if __name__ == "__main__":
    # 加载目标和参考图像
    target_path = 'images/target.jpeg'  # 替换为实际路径
    reference_path = 'images/1.jpeg'  # 替换为实际路径
    target_image = cv2.imread(target_path)
    reference_image = cv2.imread(reference_path)

    # 验证图像
    if target_image is None or reference_image is None:
        raise ValueError("Error loading images. Please check the image paths.")

    # 运行 YOLO 预测
    target_mod = get_img_mod(target_path)
    reference_mod = get_img_mod(reference_path)

    # 处理并生成每个目标结果的建议
    for target_obj in target_mod:
        # 初始化目标图像的副本以进行绘制
        target_img_with_box = target_mod.plot()
        
        # 根据关键点数量或其他标准找到最佳匹配的参考结果
        # best_match = None
        # for reference_obj in reference_mod:
            # if len(reference_obj.keypoints.data[0]) >= len(target_obj.keypoints.data[0]):
            # if target_obj.cls[0] == reference_obj.cls[0]:
            #     best_match = reference_obj
            #     break
        
        # 如果找到匹配项，则生成姿态建议
        # if best_match:
        if reference_mod:
            advise_pose(
                # target_obj, best_match, target_img_with_box,
                target_obj, reference_mod, target_img_with_box,
                target_image.shape[1], target_image.shape[0],
                reference_image.shape[1], reference_image.shape[0]
            )
            
            # 保存结果图像
            origin_image_name, origin_image_ext = os.path.splitext(os.path.basename(target_path))
            output_image_with_pose_path = f'images/p_{origin_image_name}{origin_image_ext}'
            cv2.imwrite(output_image_with_pose_path, target_img_with_box)
            print(f"Image with pose advice saved at {output_image_with_pose_path}")
        else:
            print("No suitable reference match found for the target.")
