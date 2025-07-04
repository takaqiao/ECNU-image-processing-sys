# 综合图像处理与 AI 应用系统

本项目是一个基于 Python Flask 和 OpenCV 的 Web 应用，集成了一系列从基础到高级的图像处理功能。系统不仅涵盖了经典的图像算法，如滤波、形态学和边缘检测，还整合了基于深度学习的先进 AI 应用，包括神经网络风格迁移和 YOLOv8 驱动的目标检测。

## ✨ 主要功能

### 基础图像处理

- **图像变换**: 直方图均衡化、绘制直方图、二值化。
- **几何变换**: 水平/垂直/对角翻转、仿射变换、缩放、平移、旋转。
- **空间域滤波**: 中值滤波、空域锐化。
- **边缘与线条**: Roberts、Sobel、Laplacian、Canny 算子及霍夫直线检测。
- **形态学操作**: 腐蚀、膨胀、开运算、闭运算。
- **频域处理**: 基于 FFT 的低通（平滑）和高通（锐化）滤波。
- **噪声模拟**: 添加椒盐噪声和高斯噪声。

### 高级 AI 应用

- **神经网络风格迁移**: 将图像内容与多种艺术风格（如梵高《星夜》、蒙克《呐喊》）进行融合。
- **YOLOv8 通用目标检测**: 使用预训练的 YOLOv8n 模型，识别和定位图像中的常见物体。
- **PCB 缺陷检测**: 使用在 PCB 数据集上微调的 YOLOv8 模型，自动检测电路板上的短路、开路等缺陷。
- **X 光骨折辅助检测**: 使用专门训练的 YOLOv8 模型，在 X 光片中辅助识别疑似骨折区域。

## 🛠️ 技术栈

- **后端**: Python 3.8, Flask
- **图像处理**: OpenCV-Python, Pillow, NumPy
- **深度学习**: PyTorch, Ultralytics YOLOv8
- **环境管理**: Anaconda

## 🚀 快速上手

请遵循以下步骤在您的本地机器上安装并运行此项目。

### 1. 先决条件

- 已安装 [Git](https://git-scm.com/)
- 已安装 [Anaconda](https://www.anaconda.com/download/)

### 2. 安装与配置

**a. 克隆仓库**

```bash
git clone https://github.com/takaqiao/ECNU-image-processing-sys
cd ECNU-image-processing-sys
```

**b. 创建并激活 Anaconda 环境**

这将创建一个名为 `imgpro_sys` 的独立 Python 3.8 环境，以避免依赖冲突。

```bash
conda create -n imgpro_sys python=3.8 -y
conda activate imgpro_sys
```

**c. 安装 Python 依赖**

使用项目提供的 `requirements.txt` 文件一键安装所有必需的库。

```bash
pip install -r requirements.txt
```

### 3. 启动应用

在已激活 `imgpro_sys` 环境的终端中，运行以下命令启动 Flask 应用：

```bash
flask run
```

### 4. 访问系统

应用启动后，打开您的 Web 浏览器并访问以下地址：

`http://127.0.0.1:5000`

## 🙏 致谢与模型来源

本项目的部分功能实现，特别是高级 AI 应用，得益于以下优秀的开源项目和预训练模型。我们在此表示诚挚的感谢：

- **X 光骨折检测模型**: 来源于 [ashita03/Bone-Fracture-Detection](https://github.com/ashita03/Bone-Fracture-Detection)。
- **PCB 缺陷检测模型**: 来源于 [ximenyuejun/SRI-YOLO-PCB-Defect-Detection](https://github.com/ximenyuejun/SRI-YOLO-PCB-Defect-Detection)。
- **通用目标检测模型 (YOLOv8n)**: 由 [Ultralytics](https://huggingface.co/Ultralytics/YOLOv8) 提供。
