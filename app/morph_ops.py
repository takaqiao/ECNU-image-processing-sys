import cv2

def erosion(img, ksize):
    """腐蚀操作"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(img, kernel)

def dilation(img, ksize):
    """膨胀操作"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.dilate(img, kernel)

def open_op(img, ksize):
    """开运算"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def close_op(img, ksize):
    """闭运算"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
