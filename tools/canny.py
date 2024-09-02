import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from PIL import Image


# ------------------------------#
#   image：要检测的图像
#   threshold1：阈值1（最小值）
#   threshold2：阈值2（最大值），使用此参数进行明显的边缘检测
#   edges：图像边缘信息
#   apertureSize：sobel算子（卷积核）大小
#    L2gradient ：布尔值:
#       True： 使用更精确的L2范数进行计算（即两个方向的导数的平方和再开方）
#       False：使用L1范数（直接将两个方向导数的绝对值相加）
# ------------------------------#
img = cv2.imread('img/11_mask.jpg')
edges = cv2.Canny(img,100,200)
edges2 = cv2.Canny(img,50,200)

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.subplot(132)
plt.imshow(edges,cmap = 'gray')
plt.subplot(133)
plt.imshow(edges2,cmap = 'gray')
plt.show()
#plt.savefig('./img_out/11_maskpredict.png',dpi=500)

cv2.imshow("11_maskpredict",edges)
cv2.waitKey(0)
cv2.imwrite('img_out/11_maskpredict.png', edges)


