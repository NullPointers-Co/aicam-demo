import cv2
from ultralytics import YOLO  # 替换为实际模块

def preview(weights, video_path, out_file_path, origin=False, box=False):

    # 加载预训练模型
    model = YOLO(weights)  # 替换为实际权重路径

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

    # 打开视频
    video_path = video_path  # 替换为你的输入视频路径
    cap = cv2.VideoCapture(video_path)

    # 检查是否成功打开视频
    if not cap.isOpened():
        print("无法打开视频")
        exit()

    # 获取视频帧率和分辨率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频设置
    out = cv2.VideoWriter(
        out_file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧输入模型进行预测
        results = model.predict(frame, conf=0.5, task='pose')

        # 提取关键点
        for person in results:  # 假设每人返回一个关键点列表
            box_center_x, box_center_y, _w, _h = person.boxes.xywh[0]
            box_xyxy = person.boxes.xyxy[0]
            keypoints = person.keypoints.data[0]
            for i, (x, y, conf) in enumerate(keypoints):
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                if origin:
                    cv2.circle(frame, (int(box_center_x), int(box_center_y)), 5, (0, 0, 255), -1)
                if box:
                    cv2.rectangle(frame, (int(box_xyxy[0]), int(box_xyxy[1])),
                                  (int(box_xyxy[2]), int(box_xyxy[3])),
                                  (255, 0, 0), 2)

            # 连接关键点形成骨架
            skeleton = [
                (0, 1), (1, 2), (0, 2), (1, 3), (2, 4), # 头
                (0, 5), (0, 6),  # 头到躯干
                (5, 6), (6, 12), (12, 11), (11, 5), # 躯干
                (5, 7), (7, 9), # 左臂
                (6, 8), (8, 10),  # 右臂
                (11, 13), (13, 15),  # 左腿
                (12, 14), (14, 16)  # 右腿
            ]
            for start, end in skeleton:
                if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
                    x1, y1 = keypoints[start][:2]
                    x2, y2 = keypoints[end][:2]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # 显示帧或保存到输出视频
        out.write(frame)

    cap.release()
    out.release()
