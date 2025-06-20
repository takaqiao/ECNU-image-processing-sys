import cv2
import numpy as np

def roberts_edge(img):
    """
    使用Roberts算子进行边缘检测。

    :param img: 输入的BGR图像。
    :return: 边缘检测后的灰度图像。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_x = np.array([[-1, 0], [0, 1]], dtype=int)
    kernel_y = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(gray, cv2.CV_16S, kernel_x)
    y = cv2.filter2D(gray, cv2.CV_16S, kernel_y)
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

def sobel_edge(img):
    """
    使用Sobel算子进行边缘检测。

    :param img: 输入的BGR图像。
    :return: 边缘检测后的灰度图像。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

def laplacian_edge(img):
    """
    使用Laplacian算子进行边缘检测。

    :param img: 输入的BGR图像。
    :return: 边缘检测后的灰度图像。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dst = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
    return cv2.convertScaleAbs(dst)

def canny_edge(img, thres1, thres2):
    """
    使用Canny算法进行边缘检测。

    :param img: 输入的BGR图像。
    :param thres1: 低阈值。
    :param thres2: 高阈值。
    :return: 边缘检测后的二值化图像。
    """
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.Canny(blurred, int(thres1), int(thres2))