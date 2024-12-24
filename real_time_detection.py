import cv2
import torch
import sys

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

    # 使用 YOLOv5 进行检测
    results = model(frame)  # 输入当前帧图像到模型进行推理

    # 获取检测框并在原图上绘制
    for result in results.xyxy[0]:  # 遍历每个检测结果
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf >= 0.5:  # 置信度阈值，过滤低置信度检测框
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f'{model.names[int(cls)]} {conf:.2f}'  # 标签和置信度

            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

            # 绘制标签
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示最终结果
    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
