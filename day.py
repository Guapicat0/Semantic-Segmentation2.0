import matplotlib.pyplot as plt
from skimage import io,data
from skimage.io import imread, imshow
from skimage import exposure
#调节亮度

image = imread('img/11.jpg')
image_bright = exposure.adjust_gamma(image, gamma=0.5,gain=1)
image_dark = exposure.adjust_gamma(image, gamma=11,gain=1)
# 显示图像
plt.subplot(131)
imshow(image)
plt.title('Original Image')
plt.subplot(132)
imshow(image_bright)
plt.title('Bright Image')
plt.subplot(133)
imshow(image_dark)
plt.title('Dark Image')
plt.show()
io.imsave('img/transforms/11-dark.jpg',image_dark)
