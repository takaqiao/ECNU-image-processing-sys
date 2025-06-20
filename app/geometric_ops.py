import cv2
import numpy as np

def horizontal_flip(img):
    """
    对图像进行水平翻转。

    :param img: 输入的BGR图像。
    :return: 水平翻转后的图像。
    """
    return cv2.flip(img, 1)

def vertical_flip(img):
    """
    对图像进行垂直翻转。

    :param img: 输入的BGR图像。
    :return: 垂直翻转后的图像。
    """
    return cv2.flip(img, 0)

def cross_flip(img):
    """
    对图像进行对角线（水平和垂直同时）翻转。

    :param img: 输入的BGR图像。
    :return: 对角翻转后的图像。
    """
    return cv2.flip(img, -1)

def bilinear_interpolation(img, fx, fy):
    """
    使用双线性插值进行图像缩放。

    :param img: 输入图像。
    :param fx: x轴的缩放因子。
    :param fy: y轴的缩放因子。
    :return: 缩放后的图像。
    """
    return cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

def panning(img, dx, dy):
    """
    对图像进行平移。

    :param img: 输入图像。
    :param dx: x轴方向的平移距离（像素）。
    :param dy: y轴方向的平移距离（像素）。
    :return: 平移后的图像。
    """
    height, width = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (width, height))

def rotation(img, angle, scale):
    """
    对图像进行旋转。

    :param img: 输入图像。
    :param angle: 旋转角度（度）。
    :param scale: 旋转后的缩放因子。
    :return: 旋转后的图像。
    """
    height, width = img.shape[:2]
    # 以图像中心为旋转点
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle=angle, scale=scale)
    return cv2.warpAffine(img, M, (width, height))

def affine_transform(img):
    """
    对图像应用一个预设的仿射变换。

    :param img: 输入图像。
    :return: 仿射变换后的图像。
    """
    rows, cols, _ = img.shape
    # 定义仿射变换前后的三个对应点
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows))