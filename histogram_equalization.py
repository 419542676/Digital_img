import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载原始图片
image_path = 'faces/1.jpg'
image = cv2.imread(image_path)

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用直方图均衡化提高对比度
equalized_image = cv2.equalizeHist(gray_image)

# 显示原始图像与增强后的图像
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()

# 保存增强后的图像
cv2.imwrite('faces/1_equalized.jpg', equalized_image)
