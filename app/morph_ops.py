import cv2
import numpy as np

def erosion(img, ksize):
    """
    对图像执行腐蚀操作，通常用于去除小的白色噪声点。

    :param img: 输入图像。
    :param ksize: 用于腐蚀的结构元内核大小。
    :return: 腐蚀后的图像。
    """
    ksize = int(ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(img, kernel)

def dilation(img, ksize):
    """
    对图像执行膨胀操作，可以连接邻近的白色区域。

    :param img: 输入图像。
    :param ksize: 用于膨胀的结构元内核大小。
    :return: 膨胀后的图像。
    """
    ksize = int(ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.dilate(img, kernel)

def open_op(img, ksize):
    """
    对图像执行开运算（先腐蚀后膨胀），用于去除小的噪点。

    :param img: 输入图像。
    :param ksize: 用于开运算的结构元内核大小。
    :return: 开运算后的图像。
    """
    ksize = int(ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def close_op(img, ksize):
    """
    对图像执行闭运算（先膨胀后腐蚀），用于填充前景物体中的小洞。

    :param img: 输入图像。
    :param ksize: 用于闭运算的结构元内核大小。
    :return: 闭运算后的图像。
    """
    ksize = int(ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)