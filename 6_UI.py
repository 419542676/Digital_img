import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import torch
import sys
import numpy as np
from deepface import DeepFace

# 确保 yolov5 目录在 Python 的搜索路径中
sys.path.append('yolov5')  # 如果 yolov5 文件夹和 real_time_detection.py 在同一目录下

# 从本地加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/yolov5s.pt')  # 使用本地权重文件

# 加载已保存的面部特征
A_features = np.load('A_face_features.npy', allow_pickle=True)

# 定义人物A的基本信息
A_info = {
    "name": "谢铄豪",
    "id": "2022463030133",
    "gender": "男",
    "student_number": "2022463030133",
    "school": "东莞理工学院",
    "major": "软件工程"
}

def recognize_face(face_image, known_features, threshold=4.0):
    try:
        # 确保图像为RGB格式
        if face_image.shape[2] == 1:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)

        # 提取当前检测到的人脸特征
        embedding = DeepFace.represent(face_image, model_name='Facenet', enforce_detection=False)
        if len(embedding) > 0:
            current_face_feature = embedding[0]['embedding']
        else:
            print("未能提取人脸特征")
            return False, None

        # 比对特征向量
        for known_feature in known_features:
            distance = np.linalg.norm(current_face_feature - known_feature)
            if distance < threshold:
                return True, distance
    except Exception as e:
        print(f"特征提取或比对失败：{e}")
    return False, None

def enhance_image(frame):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 应用高斯模糊，内核大小调整为(3, 3)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 将增强的灰度图像转换回BGR彩色图像
    enhanced_frame = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    return enhanced_frame

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别扫描")
        self.root.geometry("1200x800")

        # 创建标题栏
        title_label = tk.Label(root, text="人脸识别扫描", font=("Arial", 24))
        title_label.pack(pady=10)

        # 左侧区域 - 实时视频检测
        self.video_frame = tk.Label(root, bd=2, relief="solid", text="人脸识别中,请稍后…")
        self.video_frame.pack(side="left", padx=20, pady=20)

        # 右侧区域 - 显示识别出来的照片和信息
        self.info_frame = tk.Frame(root, bd=2, relief="solid")
        self.info_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # 显示识别出来的照片
        self.recognized_photo = tk.Label(self.info_frame)
        self.recognized_photo.pack(pady=10)

        # 用户姓名按钮
        self.name_button = tk.Button(self.info_frame, text="姓名", font=("Arial", 14))
        self.name_button.pack(pady=10)

        # 个人信息区
        info_label = tk.Label(self.info_frame, text="个人信息", font=("Arial", 18))
        info_label.pack(pady=10)

        self.info_table = ttk.Treeview(self.info_frame, columns=("属性", "值"), show="headings", height=5)
        self.info_table.heading("属性", text="属性")
        self.info_table.heading("值", text="值")
        self.info_table.pack(fill="both", padx=10, pady=10)

        # 开启视频流
        self.cap = cv2.VideoCapture(0)
        self.update_video_frame()

    def insert_info(self, attr, value):
        self.info_table.insert("", "end", values=(attr, value))

    # 在 update_video_frame 方法中，修改以下部分：
    def update_video_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 进行图像增强处理
            enhanced_frame = enhance_image(frame)

            # 使用 YOLOv5 进行人脸检测
            results = model(enhanced_frame)  # 输入增强后的图像到模型进行推理

            # 获取检测框并在原图上绘制
            for result in results.xyxy[0]:  # 遍历每个检测结果
                x1, y1, x2, y2, conf, cls = result[:6]
                if conf >= 0.7:  # 置信度阈值，过滤低置信度检测框
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    face_width, face_height = x2 - x1, y2 - y1

                    # 检查人脸区域大小，确保合理
                    if face_width <= 0 or face_height <= 0:
                        continue

                    # 提取人脸区域并进行比对
                    face_image = frame[y1:y2, x1:x2]  # 使用原始帧的区域进行比对
                    if face_image.size == 0:
                        continue

                    try:
                        is_A, distance_A = recognize_face(face_image, A_features)

                        if is_A:
                            label = f'{A_info["name"]} (ID: {A_info["id"]})'
                            self.update_info(A_info)
                        else:
                            label = ''  # 如果不是A，清空信息
                            self.clear_info()
                    except Exception as e:
                        print(f"比对失败：{e}")
                        label = ''  # 比对失败时，不显示任何信息

                    if label:  # 只有在识别到A时才绘制标签和框
                        # 转换为Pillow图像对象，以便使用Pillow绘制中文文本
                        pil_img = Image.fromarray(frame)
                        draw = ImageDraw.Draw(pil_img)

                        font = ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", 20)  # 微软雅黑

                        # 绘制检测框
                        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)  # 绿色框

                        # 绘制中文标签
                        draw.text((x1, y1 - 30), label, font=font, fill="green")  # 绘制文本（姓名）

                        # 将图像转回OpenCV格式
                        frame = np.array(pil_img)

            # 转换为图像格式
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.config(image=imgtk)

        self.root.after(10, self.update_video_frame)

    def update_info(self, person_info):
        self.clear_info()
        self.insert_info("姓名", person_info["name"])
        self.insert_info("性别", person_info["gender"])
        self.insert_info("学号", person_info["student_number"])
        self.insert_info("学校", person_info["school"])
        self.insert_info("专业", person_info["major"])

    def clear_info(self):
        for item in self.info_table.get_children():
            self.info_table.delete(item)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

# 创建主窗口
root = tk.Tk()
app = FaceRecognitionApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
