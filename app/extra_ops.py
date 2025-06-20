import cv2
import numpy as np
import matplotlib
import io

# 设置matplotlib后端，防止在无界面的服务器上出错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
#  1. 图像变换与增强 (Transforms & Enhancement)
# ==============================================================================

def binarize(img, threshold):
    """图像二值化处理"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # 转换回3通道以便在彩色界面上显示
    return cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

def plot_histogram(img):
    """绘制并返回灰度直方图图像"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- 关键修正：使用更稳健的方式将matplotlib绘图转换为OpenCV图像 ---
    fig = plt.figure(figsize=(6, 5)) # 创建一个新的图形
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.plot(cv2.calcHist([gray], [0], None, [256], [0, 256]), color='teal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0, 256])
    
    # 1. 将图形保存到内存缓冲区
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # 2. 从缓冲区读取数据并用OpenCV解码
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    hist_img_bgr = cv2.imdecode(img_arr, 1)
    
    # 3. 关闭图形以释放内存
    plt.close(fig)
    
    return hist_img_bgr

# ==============================================================================
#  2. 空间域滤波 (Spatial Filtering)
# ==============================================================================

def median_filter(img, ksize):
    """中值滤波"""
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)

def sharpen(img):
    """空域锐化 (使用拉普拉斯算子)"""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

# ==============================================================================
#  3. 噪声生成 (Noise Generation)
# ==============================================================================

def add_salt_and_pepper_noise(img, amount):
    """添加椒盐噪声"""
    output = img.copy()
    num_pixels = int(amount * img.size / 3) # 每个像素有3个通道
    
    # Salt noise
    salt_coords = [np.random.randint(0, i - 1, num_pixels) for i in img.shape[:2]]
    output[salt_coords[0], salt_coords[1]] = (255, 255, 255)

    # Pepper noise
    pepper_coords = [np.random.randint(0, i - 1, num_pixels) for i in img.shape[:2]]
    output[pepper_coords[0], pepper_coords[1]] = (0, 0, 0)
    return output
    
def add_gaussian_noise(img, mean=0, var=0.01):
    """添加高斯噪声"""
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, img.shape) * 255
    noisy_img = img.astype(np.float32) + gaussian
    np.clip(noisy_img, 0, 255, out=noisy_img)
    return noisy_img.astype(np.uint8)

# ==============================================================================
#  4. 频域处理 (Frequency Domain)
# ==============================================================================

def _fft_process(img, mask):
    """通用的FFT处理流程"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    fshift = dft_shift * mask
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def fft_lowpass(img, radius):
    """频域低通滤波（平滑）"""
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), int(radius), (1,1), -1)
    return _fft_process(img, mask)

def fft_highpass(img, radius):
    """频域高通滤波（锐化）"""
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), int(radius), (0,0), -1)
    return _fft_process(img, mask)

# ==============================================================================
#  5. 其他检测 (Other Detections)
# ==============================================================================

def hough_transform(img):
    """霍夫直线检测"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    
    result_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return result_img
