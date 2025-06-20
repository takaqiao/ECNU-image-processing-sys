# 图像处理与 AI 应用系统

## 项目简介

本项目旨在构建一个集成了基础图像处理功能、高级 AI 能力（如目标检测）以及特定应用场景解决方案（PCB 缺陷检测、AI 病历分析）的图像处理系统。

## 技术栈

- **后端:** Python 3.8+, Flask
- **基础图像处理:** OpenCV-Python, Pillow
- **高级 AI:** PyTorch, Ultralytics YOLOv8, Tesseract, pytesseract, spaCy, medspaCy
- **前端:** HTML, CSS

## 环境配置

1.  安装 Anaconda: [https://www.anaconda.com/download/](https://www.anaconda.com/download/)
2.  创建并激活虚拟环境:
    \`\`\`bash
    conda create -n imgpro_sys python=3.8 -y
    conda activate imgpro_sys
    \`\`\`
3.  安装依赖:
    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`
4.  安装 Tesseract OCR (如果需要病历分析功能): [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html)
5.  下载 YOLOv8 预训练模型 (`yolov8n.pt`) 并放置在 `models/` 目录下: [https://github.com/ultralytics/yolov5/releases/tag/v8.0](https://github.com/ultralytics/yolov5/releases/tag/v8.0) (选择 `yolov8n.pt`)

## 运行

1.  在项目根目录下执行:
    \`\`\`bash
    flask run
    \`\`\`
2.  在浏览器中访问 `http://127.0.0.1:5000`。

## 功能

- **基础图像处理:**
  - 灰度转换
  - 高斯模糊
  - 直方图均衡化
- **高级功能:**
  - YOLOv8 目标检测
- **实际应用:**
  - PCB 缺陷检测 (占位符)
  - AI 病历分析 (占位符)

## 注意

- PCB 缺陷检测和 AI 病历分析的具体模型训练和数据准备需要用户自行完成。
- 本代码仅为基本框架，具体图像处理算法需要在相应的 Python 文件中实现。

## 使用到的开源项目

- Flask: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
- OpenCV-Python: [https://opencv.org/](https://opencv.org/)
- Pillow: [https://python-pillow.org/](https://python-pillow.org/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- Ultralytics YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Tesseract OCR: [https://tesseract-ocr.github.io/](https://tesseract-ocr.github.io/)
- pytesseract: [https://pypi.org/project/pytesseract/](https://pypi.org/project/pytesseract/)
- spaCy: [https://spacy.io/](https://spacy.io/)
- medspaCy: [https://allenai.github.io/medspacy/](https://allenai.github.io/medspacy/)
