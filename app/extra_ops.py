import cv2
import numpy as np
import matplotlib
import io

# 设置matplotlib后端为'Agg'，避免在无GUI的服务器环境中因尝试创建UI窗口而报错。
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def binarize(img, threshold):
    """
    对图像进行二值化处理。

    :param img: 输入图像。
    :param threshold: 用于二值化的阈值 (0-255)。
    :return: 二值化后的BGR图像。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR) # 转回BGR以统一格式

def plot_histogram(img):
    """
    计算并绘制图像的灰度直方图，将绘图结果作为图像返回。

    :param img: 输入图像。
    :return: 包含直方图的BGR图像。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- 关键步骤：使用内存缓冲区将matplotlib绘图转换为OpenCV图像 ---
    # 这是最稳健的方法，因为它不依赖于特定的GUI后端，在服务器和本地都能良好工作。
    fig = plt.figure(figsize=(6, 5)) 
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.plot(cv2.calcHist([gray], [0], None, [256], [0, 256]), color='teal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0, 256])
    
    # 1. 将图形保存到内存IO流中
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # 2. 从IO流中读取数据并用OpenCV解码为图像
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    hist_img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    
    # 3. 关闭图形以释放内存，防止内存泄漏
    plt.close(fig)
    
    return hist_img_bgr

def median_filter(img, ksize):
    """
    应用中值滤波以去除噪声。

    :param img: 输入图像。
    :param ksize: 滤波器内核大小，必须是奇数。
    :return: 滤波后的图像。
    """
    ksize = int(ksize)
    # 内核大小必须是奇数，如果输入为偶数，则自动加1
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)

def sharpen(img):
    """
    使用拉普拉斯算子核进行图像锐化。

    :param img: 输入图像。
    :return: 锐化后的图像。
    """
    # 这个核增强了中心像素，同时抑制了周围像素，从而达到锐化效果。
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def add_salt_and_pepper_noise(img, amount):
    """
    为图像添加椒盐噪声。

    :param img: 输入图像。
    :param amount: 噪声比例 (0.0 to 1.0)。
    :return: 添加噪声后的图像。
    """
    output = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    output[coords[0], coords[1], :] = 255

    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    output[coords[0], coords[1], :] = 0
    return output
    
def add_gaussian_noise(img, mean=0, var=0.01):
    """
    为图像添加高斯噪声。

    :param img: 输入图像。
    :param mean: 噪声均值。
    :param var: 噪声方差。
    :return: 添加噪声后的图像。
    """
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, img.shape) * 255
    noisy_img = np.clip(img.astype(np.float32) + gaussian, 0, 255)
    return noisy_img.astype(np.uint8)

def _fft_process(img, mask):
    """
    通用的频域处理流程，封装了FFT、掩码应用和IFFT的通用步骤。

    :param img: 输入图像。
    :param mask: 应用于频域的掩码。
    :return: 经过频域处理后的BGR图像。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. 傅里叶变换
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 2. 将零频率分量移到中心
    dft_shift = np.fft.fftshift(dft)
    # 3. 应用掩码 (低通/高通)
    fshift = dft_shift * mask
    # 4. 逆向移位
    f_ishift = np.fft.ifftshift(fshift)
    # 5. 傅里叶逆变换
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    # 6. 标准化到0-255范围以便显示
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def fft_lowpass(img, radius):
    """
    频域低通滤波（平滑）。

    :param img: 输入图像。
    :param radius: 低通滤波的半径。
    :return: 滤波后的图像。
    """
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), int(radius), (1, 1), -1) # 创建一个中心为1，其余为0的圆形掩码
    return _fft_process(img, mask)

def fft_highpass(img, radius):
    """
    频域高通滤波（锐化）。

    :param img: 输入图像。
    :param radius: 高通滤波的半径（中心被抑制的区域）。
    :return: 滤波后的图像。
    """
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), int(radius), (0, 0), -1) # 创建一个中心为0，其余为1的圆形掩码
    return _fft_process(img, mask)

def hough_transform(img):
    """
    使用霍夫变换进行直线检测。

    :param img: 输入图像。
    :return: 绘制了检测到的直线的图像。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 使用概率霍夫变换，效率更高
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    
    result_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return result_img