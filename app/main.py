import os
import cv2
import base64
import numpy as np
import traceback
from PIL import Image

# --- Flask App 及路由 ---
from app import app
from flask import render_template, request, jsonify

# --- AI/ML 模型库 ---
from ultralytics import YOLO

# --- 自定义图像处理模块 ---
from app.geometric_ops import (horizontal_flip, vertical_flip, cross_flip,
                            bilinear_interpolation, panning, rotation, affine_transform)
from app.edge_ops import roberts_edge, sobel_edge, laplacian_edge, canny_edge
from app.morph_ops import erosion, dilation, open_op, close_op
from app.extra_ops import (binarize, plot_histogram, median_filter, sharpen,
                           add_salt_and_pepper_noise, add_gaussian_noise,
                           fft_lowpass, fft_highpass, hough_transform)


# ==============================================================================
#  模型加载 (在应用启动时执行一次)
# ==============================================================================

def load_yolo_model(model_path):
    """ 安全地加载YOLO模型，如果文件不存在则返回None。 """
    try:
        if os.path.exists(model_path):
            return YOLO(model_path)
        print(f"警告: 模型文件未找到于 '{model_path}'")
    except Exception as e:
        print(f"加载模型 '{model_path}' 时出错: {e}")
    return None

# --- 加载所有应用所需的模型 ---
# 备注: 这些模型需要用户根据README预先下载或训练。
yolo_model = load_yolo_model('models/yolov8n.pt')
pcb_model = load_yolo_model('models/pcb_defect_model.pt')
fracture_model = load_yolo_model('models/bone_fracture_model.pt')


# ==============================================================================
#  高级 AI 功能
# ==============================================================================

def transfer_style(img, style_name):
    """
    应用神经网络风格迁移。

    :param img: 输入的OpenCV图像 (BGR格式)。
    :param style_name: 风格模型的名称 (不含扩展名)。
    :return: 风格迁移后的图像。
    :raises FileNotFoundError: 如果对应的风格模型文件不存在。
    """
    model_path = os.path.join('models/style_models', f'{style_name}.t7')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"风格模型文件未找到: {model_path}")
    
    net = cv2.dnn.readNetFromTorch(model_path)
    h, w = img.shape[:2]
    
    # 图像预处理：减去ImageNet的均值，这是模型训练时所用的预处理步骤
    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    
    net.setInput(blob)
    out = net.forward()
    
    # 后处理：调整输出维度，并加上之前减去的均值
    out = out.reshape(3, out.shape[2], out.shape[3]).transpose(1, 2, 0)
    out += (103.939, 116.779, 123.680)
    out = np.clip(out, 0, 255).astype('uint8')
    return out


def detect_with_yolo(img, model, custom_label=None):
    """
    通用的YOLOv8检测函数，用于在图像上检测对象并绘制标注。

    :param img: 输入图像。
    :param model: 已加载的YOLO模型。
    :param custom_label: (可选) 如果提供，则所有检测框都使用此标签。
    :return: 一个包含标注后图像和检测信息的字典。
    :raises FileNotFoundError: 如果模型未成功加载。
    """
    if model is None:
        model_name = [k for k,v in globals().items() if v is model][0]
        raise FileNotFoundError(f"YOLO模型 '{model_name}' 未被成功加载。请检查模型文件路径。")

    results = model(img)
    annotated_img = img.copy()
    detections = []
    
    h, w, _ = annotated_img.shape
    base_size = max(h, w)
    
    # --- 根据图像尺寸动态调整标注样式，以获得更好的视觉效果 ---
    line_thickness = max(1, int(base_size / 400))
    font_scale = max(0.5, base_size / 1200)
    font_thickness = max(1, int(font_scale * 2))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            
            name = custom_label if custom_label else model.names[cls_id]
            label = f'{name} {conf:.2f}'
            detections.append({"class": name, "confidence": conf, "box": [x1, y1, x2, y2]})

            # --- 绘制检测框和标签背景 ---
            color = (0, 255, 0) if custom_label is None else ((255, 0, 0) if custom_label == 'fracture' else (0, 0, 255))
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, line_thickness)
            
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # 智能判断标签位置，防止超出图像顶部
            label_y = y1 - 10 if y1 - label_h - 10 > 0 else y1 + label_h + 10
            
            cv2.putText(annotated_img, label, (x1, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    return {"image": annotated_img, "detections": detections}


# ==============================================================================
#  辅助函数
# ==============================================================================

def image_to_base64(img):
    """ 将OpenCV图像编码为Base64字符串以便在HTML中显示。 """
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('ascii')

def preprocess_image(img, max_size=1280):
    """ 预处理图像，如果尺寸过大则进行缩放，以提高处理效率。 """
    h, w = img.shape[:2]
    if h > max_size or w > max_size:
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


# ==============================================================================
#  Flask 路由
# ==============================================================================

@app.route('/')
def index():
    """ 渲染主页面。 """
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def handle_processing():
    """
    处理图像的核心API接口。
    接收图片和操作指令，返回处理结果。
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': '请求中没有图像文件'}), 400
        
        file = request.files['image']
        operation = request.form.get('operation')

        if not operation:
            return jsonify({'error': '未指定操作'}), 400
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        # --- 图像解码和预处理 ---
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = preprocess_image(img)

        # --- 解析操作参数 ---
        params = {k: v for k, v in request.form.items() if k != 'operation'}
        for key, value in params.items():
            try:
                # 尝试将参数转换为数值类型
                params[key] = float(value) if '.' in str(value) else int(value)
            except (ValueError, TypeError):
                pass
        
        # ======================================================================
        #  操作分派器 (REFACTORED)
        #  使用字典统一管理所有操作，增强可维护性和可扩展性。
        # ======================================================================
        OPERATIONS_MAP = {
            # --- 基础操作 ---
            'histogram_equalization': lambda i, **p: cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR),
            'plot_histogram': plot_histogram,
            'sharpen': sharpen,
            'add_gaussian_noise': add_gaussian_noise,
            'hough_transform': hough_transform,
            'horizontal_flip': horizontal_flip, 
            'vertical_flip': vertical_flip, 
            'cross_flip': cross_flip,
            'affine_transform': affine_transform,
            'roberts_edge': roberts_edge, 
            'sobel_edge': sobel_edge, 
            'laplacian_edge': laplacian_edge,
            # --- 带参数操作 ---
            'binarize': binarize,
            'median_filter': median_filter,
            'add_salt_and_pepper_noise': add_salt_and_pepper_noise,
            'fft_lowpass': fft_lowpass,
            'fft_highpass': fft_highpass,
            'bilinear_interpolation': bilinear_interpolation, 
            'panning': panning, 
            'rotation': rotation,
            'canny_edge': canny_edge, 
            'erosion': erosion, 
            'dilation': dilation, 
            'open_op': open_op, 
            'close_op': close_op,
        }
        
        processed_data = {}

        if operation in OPERATIONS_MAP:
            result_img = OPERATIONS_MAP[operation](img, **params)
            processed_data = {'image': image_to_base64(result_img)}
            
        elif operation.startswith('style_'):
            style_name = operation.split('_', 1)[1]
            result_img = transfer_style(img, style_name=style_name)
            processed_data = {'image': image_to_base64(result_img)}
            
        elif operation == 'yolov8_detect':
            result = detect_with_yolo(img, yolo_model)
            processed_data = {'image': image_to_base64(result['image']), 'detections': result['detections']}
            
        elif operation == 'pcb_defect_check':
            result = detect_with_yolo(img, pcb_model)
            processed_data = {'image': image_to_base64(result['image']), 'detections': result['detections']}
            
        elif operation == 'bone_fracture_detect':
            result = detect_with_yolo(img, fracture_model, custom_label='fracture')
            processed_data = {'image': image_to_base64(result['image']), 'detections': result['detections']}
            
        else:
            return jsonify({'error': f'未知操作: {operation}'}), 400

        return jsonify(processed_data)

    except FileNotFoundError as e:
        # 捕获模型文件未找到的特定错误
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        # 捕获所有其他异常，并打印traceback以便调试
        traceback.print_exc()
        return jsonify({'error': f'服务器处理时发生意外错误: {str(e)}'}), 500