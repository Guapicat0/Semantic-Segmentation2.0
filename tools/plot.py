import numpy as np
import cv2
from PIL import Image
import random

np.set_printoptions(threshold=np.inf)
color=[0,255,255] #BGR
img1 = Image.open('img_out/11_maskpredict.png')
img1= np.array(img1)
img2 = Image.open('img/11.jpg')
img2= np.array(img2)
# way2
#img2 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)

image_R=img1
image_G=img1
image_B=img1
image_R=np.where(image_R==255,color[0],0)
image_G=np.where(image_G==255,color[1],0)
image_B=np.where(image_B==255,color[2],0)
stacked_img = np.stack((image_R,image_G,image_B), axis=2)
img1   = Image.fromarray(np.uint8(stacked_img))
img2   = Image.fromarray(np.uint8(img2))
image   = Image.blend(img1, img2, 0.5)
#读取图片
image =np.array(image)
cv2.imwrite('2.jpg',image)
s = cv2.imread('2.jpg')
cv2.imshow('img2',s)
cv2.waitKey(0)



