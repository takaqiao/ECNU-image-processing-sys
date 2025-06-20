import cv2
import numpy as np

# --- 翻转操作 ---
def horizontal_flip(img):
    """水平翻转"""
    return cv2.flip(img, 1)

def vertical_flip(img):
    """垂直翻转"""
    return cv2.flip(img, 0)

def cross_flip(img):
    """对角翻转"""
    return cv2.flip(img, -1)

# --- 缩放操作 ---
def bilinear_interpolation(img, fx, fy):
    """双线性插值缩放"""
    return cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

# --- 平移操作 ---
def panning(img, dx, dy):
    """图像平移"""
    height, width = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (width, height))

# --- 旋转操作 ---
def rotation(img, angle, scale):
    """图像旋转"""
    height, width = img.shape[:2]
    # 以图像中心为旋转点
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle=angle, scale=scale)
    return cv2.warpAffine(img, M, (width, height))

# --- 仿射变换 ---
def affine_transform(img):
    """仿射变换"""
    rows, cols = img.shape[:2]
    post1 = np.float32([[50, 50], [200, 50], [50, 200]])
    post2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(post1, post2)
    return cv2.warpAffine(img, M, (cols, rows)) # 注意这里用 (cols, rows)
