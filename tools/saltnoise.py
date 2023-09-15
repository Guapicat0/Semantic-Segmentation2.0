import numpy as np
import cv2


def salt_pepper_noise(image, prob):
    """
    添加椒盐噪声
    :param image: 输入图像
    :param prob: 噪声比
    :return: 带有椒盐噪声的图像
    """
    salt = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                salt[i][j] = 0
            elif rdn > thres:
                salt[i][j] = 255
            else:
                salt[i][j] = image[i][j]
    return salt

if __name__ == '__main__':
   src = cv2.imread(r'E:\cv_cnn\Semantic Segmentation2.0\img\11.jpg')
   cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
   cv2.imshow('input_image', src)
   tar = salt_pepper_noise(src, 0.05)
   cv2.imshow('noise', tar)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
