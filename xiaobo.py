import matplotlib.pyplot as plt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import skimage.io
import numpy as np


def waveletDenoiseGray(src):
    # 读取图片，并转换成flaot类型
    img = skimage.io.imread(src)
    img = skimage.img_as_float(img)

    # 往图片添加随机噪声
    sigma = 0.1
    imgNoise = random_noise(img, var=sigma ** 2,mode="s&p")

    # 估计当前的图像的噪声的方差
    # 由于随机噪声的裁切，估计的sigma值将小于指定的sigma的值
    sigma_est = estimate_sigma(imgNoise, average_sigmas=True)

    # 对图像分别使用Bayesshink算法和Visushrink算法
    # 输入带噪图像，小波变换模式选择 ，阈值模式，小波分解的级别，小波基，
    imgBayes = denoise_wavelet(imgNoise, method='BayesShrink', mode='soft',
                               wavelet_levels=3, wavelet='bior6.8',
                               rescale_sigma=True)

    imgVisushrink = denoise_wavelet(imgNoise, method='VisuShrink', mode='soft',
                                    sigma=sigma_est / 3, wavelet_levels=5,
                                    wavelet='bior6.8', rescale_sigma=True)

    # 计算输入和输出之间的PSNR值
    psnrNoise = SSIM(img, imgNoise)
    psnrBayes = SSIM(img, imgBayes)
    psnrVisushrink = SSIM(img, imgVisushrink)

    # 将降噪图片结果输出出来
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(imgNoise, cmap=plt.cm.gray)
    plt.title('Noise Image')

    plt.subplot(2, 2, 3)
    plt.imshow(imgBayes, cmap=plt.cm.gray)
    plt.title('Denoised Image using Bayes')

    plt.subplot(2, 2, 4)
    plt.imshow(imgVisushrink, cmap=plt.cm.gray)
    plt.title('Denoised Image using Visushrink')

    plt.show()

    # 将PSNR的值打印处理
    print('estimate sigma:', sigma_est)
    print('PSNR[orignal vs NoiseImgae]:', psnrNoise)
    print('PSNR[orignal vs Denoise[Bayes]]:', psnrBayes)
    print('PSNR[orignal vs Denoise[Visushrink]]:', psnrVisushrink)

def waveletDenoiseRgb(src):
    # 读取图片，并转换成flaot类型
    img = skimage.io.imread(src)
    img = skimage.img_as_float(img)

    # 往图片添加随机噪声
    sigma = 0.15
    imgNoise = random_noise(img,mode='s&p')

    # 估计当前的图像的噪声的方差
    # 由于随机噪声的裁切，估计的sigma值将小于指定的sigma的值,彩色图片需要设定多通道
    sigma_est = estimate_sigma(imgNoise, multichannel=True, average_sigmas=True)

    # 对图像分别使用Bayesshink算法和Visushrink算法
    # 输入带噪图像，小波变换模式选择 ，阈值模式，小波分解的级别，小波基，
    imgBayes = denoise_wavelet(img, method='BayesShrink', mode='soft',
                               wavelet_levels=3, wavelet='coif5',
                               multichannel=True, convert2ycbcr=True,
                               rescale_sigma=True)

    imgVisushrink = denoise_wavelet(img, method='VisuShrink', mode='soft',
                                    sigma=sigma_est / 3, wavelet_levels=5,
                                    wavelet='coif5',
                                    multichannel=True, convert2ycbcr=True, rescale_sigma=True)
    # 计算输入和输出之间的PSNR值
    psnrNoise = PSNR(img, imgNoise)
    psnrBayes = PSNR(img, imgBayes)
    psnrVisushrink = PSNR(img, imgVisushrink)

    # 将降噪图片结果输出出来
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(imgNoise, cmap=plt.cm.gray)
    plt.title('Noise Image')

    plt.subplot(2, 2, 3)
    plt.imshow(imgBayes, cmap=plt.cm.gray)
    plt.title('Denoised Image using Bayes')


    plt.subplot(2, 2, 4)
    plt.imshow(imgVisushrink, cmap=plt.cm.gray)
    plt.title('Denoised Image using Visushrink')

    plt.show()

    # 将PSNR的值打印处理
    print('estimate sigma:', sigma_est)
    print('PSNR[orignal vs NoiseImgae]:', psnrNoise)
    print('PSNR[orignal vs Denoise[Bayes]]:', psnrBayes)
    print('PSNR[orignal vs Denoise[Visushrink]]:', psnrVisushrink)
    skimage.io.imsave('img/data1/101-noise-pepper.jpg', imgNoise)
    skimage.io.imsave('img/data1/101-antibayes-pepper.jpg',imgBayes)
    skimage.io.imsave('img/data1/101-antivisu-pepper.jpg', imgVisushrink)


if __name__ == "__main__":
    inputSrc = 'img/102.jpg'
    waveletDenoiseRgb(inputSrc)

