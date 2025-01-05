import os
import cv2
import numpy as np
from deepface import DeepFace

def capture_photos_from_camera(num_photos, save_folder='captured_photos/'):
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return []

    photo_count = 0
    photo_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break

        # 显示视频流
        cv2.imshow('Capture Photos', frame)

        # 等待按下 'c' 键开始拍照
        key = cv2.waitKey(1)
        if key == ord('c'):  # 按 'c' 键开始拍照
            for i in range(num_photos):
                photo_count += 1
                photo_path = os.path.join(save_folder, f"photo_{photo_count}.jpg")
                cv2.imwrite(photo_path, frame)  # 保存拍摄的照片
                print(f"照片 {photo_count} 已保存：{photo_path}")
                photo_paths.append(photo_path)

                # 再次读取视频帧，以确保每张照片不一样
                ret, frame = cap.read()
                if not ret:
                    print("无法读取视频帧")
                    break

            break  # 拍摄完成后退出循环

        # 按 'q' 键退出
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return photo_paths

def extract_features_from_photos(photo_paths):
    features = []
    for photo_path in photo_paths:
        # 提取面部特征
        try:
            if os.path.exists(photo_path):  # 确认照片文件存在
                embedding = DeepFace.represent(photo_path, model_name='Facenet', enforce_detection=False)
                face_feature = embedding[0]['embedding']
                features.append(face_feature)
                print(f"提取特征成功：{photo_path}")
            else:
                print(f"文件不存在：{photo_path}")
        except Exception as e:
            print(f"提取特征失败：{photo_path} 错误：{e}")
    return features

def save_features(features, save_path='A_face_features.npy'):
    # 保存所有特征到.npy文件
    np.save(save_path, np.array(features))
    print(f"A的所有面部特征已保存至 {save_path}")

# 拍摄 A 的多张照片并提取特征
num_photos = 200  # 拍摄的照片数量
photo_paths = capture_photos_from_camera(num_photos)

# 提取这些照片的面部特征
features = extract_features_from_photos(photo_paths)

# 将特征保存到文件
save_features(features, 'A_face_features.npy')
