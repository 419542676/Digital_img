import cv2
import torch
import sys
#使用yolo进行实时检测，并使用相关的图像增强方式进行处理对比。
# 确保 yolov5 目录在 Python 的搜索路径中
sys.path.append('yolov5')  # 如果 yolov5 文件夹和 real_time_detection.py 在同一目录下

# 从本地加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/yolov5s.pt')  # 使用本地权重文件

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 使用第一个摄像头

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()  # 读取每一帧图像
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # 拷贝一份未增强处理的帧用于显示和检测
    frame_original = frame.copy()

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # 应用高斯模糊，内核大小调整为(3, 3)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

    # 使用 YOLOv5 进行检测（未增强处理的帧）
    results_original = model(frame_original)  # 输入原始帧图像到模型进行推理

    # 使用 YOLOv5 进行检测（增强处理的帧）
    results_enhanced = model(blurred)  # 输入增强后的帧图像到模型进行推理

    # 在未增强处理的帧上绘制检测框
    for result in results_original.xyxy[0]:  # 遍历每个检测结果
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf >= 0.5:  # 置信度阈值，过滤低置信度检测框
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f'{model.names[int(cls)]} {conf:.2f}'  # 标签和置信度

            # 绘制检测框
            cv2.rectangle(frame_original, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

            # 绘制标签
            cv2.putText(frame_original, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 在增强处理的帧上绘制检测框
    for result in results_enhanced.xyxy[0]:  # 遍历每个检测结果
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf >= 0.5:  # 置信度阈值，过滤低置信度检测框
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f'{model.names[int(cls)]} {conf:.2f}'  # 标签和置信度

            # 绘制检测框
            cv2.rectangle(blurred, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

            # 绘制标签
            cv2.putText(blurred, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示未增强处理的帧检测结果
    cv2.imshow('Original Frame Detection', frame_original)

    # 显示增强处理的帧检测结果
    cv2.imshow('Enhanced Frame Detection', blurred)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
