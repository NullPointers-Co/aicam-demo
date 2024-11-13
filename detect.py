import os

import cv2
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('weights/yolov11n.pt')

def format_output(result):
    if result.boxes:
        for box in result.boxes:
            # 获取边界框的坐标
            x1, y1, x2, y2 = box.xyxy[0]  # 左上角和右下角坐标
            confidence = box.conf[0]  # 置信度
            class_idx = box.cls[0]  # 类别索引
            
            # 获取类别名称
            class_name = result.names[int(class_idx)]
            
            # 输出检测结果
            print(f"Class: {class_name}, Confidence: {confidence}, Box: [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("No objects detected.")

def get_img_mod(image_path, conf=0.35):
    _img = cv2.imread(image_path)
    _img_model = model.predict(_img, conf=conf)
    img_mod = _img_model[0]
    return img_mod

def normalize_coordinates(x1, y1, x2, y2, width, height):
    # 归一化坐标
    return x1 / width, y1 / height, x2 / width, y2 / height

def denormalize_coordinates(x1, y1, x2, y2, width, height):
    # 反归一化坐标
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)


if __name__ == "__main__":
    # 定义目标图像和参考图像的路径
    target_path = 'images/target.jpeg'
    reference_path = 'images/reference.jpeg'
    
    # 获取目标图像和参考图像的检测结果
    target_mod = get_img_mod(target_path, conf=0.35)
    reference_mod = get_img_mod(reference_path, conf=0.35)

    # 绘制目标图像的检测框
    target_img_with_box = target_mod.plot()
    # reference_img_with_box = reference_mod.plot()

    # 获取目标图像和参考图像的原始尺寸
    target_width, target_height = target_mod.orig_shape
    reference_width, reference_height = reference_mod.orig_shape

    # 打印检测结果
    print("Target:")
    format_output(target_mod)
    print("Reference:")
    format_output(reference_mod)

    # 比较目标图像和参考图像中的检测框
    for target_obj in target_mod.boxes:
        for reference_obj in reference_mod.boxes:
            if target_obj.cls[0] == reference_obj.cls[0]:
                # 归一化坐标
                target_x1, target_y1, target_x2, target_y2 = normalize_coordinates(*target_obj.xyxy[0], target_width, target_height)
                reference_x1, reference_y1, reference_x2, reference_y2 = normalize_coordinates(*reference_obj.xyxy[0], reference_width, reference_height)

                # 打印检测框的坐标和移动距离
                print(f"Target Box: [{target_x1}, {target_y1}, {target_x2}, {target_y2}]")
                print(f"Reference Box: [{reference_x1}, {reference_y1}, {reference_x2, reference_y2}]")
                print(f"x1 move: {reference_x1 - target_x1}", f"y1 move: {reference_y1 - target_y1}",
                      f"x2 move: {reference_x2 - target_x2}", f"y2 move: {reference_y2 - target_y2}")

                # 反归一化坐标
                retangle_x1, retangle_y1, retangle_x2, retangle_y2 = denormalize_coordinates(reference_x1, reference_y1, reference_x2, reference_y2, target_width, target_height)
                
                # 在目标图像上绘制绿色框
                cv2.rectangle(target_img_with_box,
                              (retangle_x1, retangle_y1), (retangle_x2, retangle_y2),
                              (0, 255, 0), 2)  # 绿色框，线条宽度为2

                # 保存带绿色框的图像
                origin_image_name, origin_image_ext = os.path.splitext(os.path.basename(target_path))
                output_image_with_box_path = f'images/d_{origin_image_name}{origin_image_ext}'
                cv2.imwrite(output_image_with_box_path, target_img_with_box)

                # 如果你想要显示图像
                # cv2.imshow("Detection Result", target_img_with_box)  
                # cv2.waitKey(0)  # 等待按键关闭窗口
                # cv2.destroyAllWindows()

            else:
                print(f"Match loss: {target_obj.cls[0]}")
