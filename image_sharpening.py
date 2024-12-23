import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载原始图片
image_path = 'faces/1.jpg'
image = cv2.imread(image_path)

# 创建锐化卷积核
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# 应用锐化卷积核
sharpened_image = cv2.filter2D(image, -1, kernel)

# 显示原始图像与锐化后的图像
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sharpened Image')
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

# 保存锐化后的图像
cv2.imwrite('faces/1_sharpened.jpg', sharpened_image)
