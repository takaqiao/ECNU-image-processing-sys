import os
import cv2
import base64
import numpy as np
import traceback
from PIL import Image

# --- Flask App Initialization ---
from app import app
from flask import render_template, request, jsonify

# --- High-Level Libraries ---
from ultralytics import YOLO

# --- Basic Operation Modules ---
from app.geometric_ops import (horizontal_flip, vertical_flip, cross_flip,
                            bilinear_interpolation, panning, rotation, affine_transform)
from app.edge_ops import roberts_edge, sobel_edge, laplacian_edge, canny_edge
from app.morph_ops import erosion, dilation, open_op, close_op
# --- 新增：导入我们的新功能模块 ---
from app.extra_ops import (binarize, plot_histogram, median_filter, sharpen,
                           add_salt_and_pepper_noise, add_gaussian_noise,
                           fft_lowpass, fft_highpass, hough_transform)


# ==============================================================================
#  BEGIN: CONSOLIDATED ADVANCED FUNCTIONS
# ==============================================================================

# --- 1. Style Transfer Function ---
def transfer_style(img, style_name):
    """Applies neural style transfer to an image."""
    model_path = os.path.join('models/style_models', f'{style_name}.t7')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"风格模型文件未找到: {model_path}")
    net = cv2.dnn.readNetFromTorch(model_path)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3]).transpose(1, 2, 0)
    out += (103.939, 116.779, 123.680)
    out = np.clip(out, 0, 255).astype('uint8')
    return out

# --- 2. YOLOv8 Object Detection ---
YOLO_MODEL_PATH = 'models/yolov8n.pt'
try:
    yolo_model = YOLO(YOLO_MODEL_PATH) if os.path.exists(YOLO_MODEL_PATH) else None
except Exception as e:
    print(f"加载YOLOv8模型时出错: {e}")
    yolo_model = None

def detect_objects(img):
    """
    使用YOLOv8模型检测物体，并使用优化的样式进行标注。
    """
    if yolo_model is None:
        raise FileNotFoundError(f"YOLOv8模型未找到: {YOLO_MODEL_PATH}")
    
    results = yolo_model(img)
    annotated_img = img.copy()
    
    h, w, _ = annotated_img.shape
    base_size = max(h, w)
    line_thickness = max(1, int(base_size / 500))
    font_scale = max(0.5, base_size / 1000)
    font_thickness = max(1, int(base_size / 400)) 
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            label = f'{yolo_model.names[cls_id]} {conf:.2f}'
            
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
            
            label_bg_y1 = y1 - label_height - baseline - 10
            label_bg_y2 = y1
            
            if label_bg_y1 < 0:
                label_bg_y1 = y1 + 10
                label_bg_y2 = y1 + label_height + baseline + 10

            sub_img = annotated_img[label_bg_y1:label_bg_y2, x1:x1 + label_width]
            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
            annotated_img[label_bg_y1:label_bg_y2, x1:x1 + label_width] = res

            text_y = label_bg_y2 - baseline - 5
            cv2.putText(annotated_img, label, (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
    return annotated_img

# --- 3. PCB Defect Analysis ---
PCB_MODEL_PATH = 'models/pcb_defect_model.pt'
try:
    pcb_model = YOLO(PCB_MODEL_PATH) if os.path.exists(PCB_MODEL_PATH) else None
except Exception as e:
    print(f"加载PCB缺陷模型时出错: {e}")
    pcb_model = None

def check_pcb_defects(img):
    """Checks for PCB defects using a custom-trained model."""
    if pcb_model is None:
        text = f"PCB Model not found at '{PCB_MODEL_PATH}'"
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return {"image": img, "detections": []}

    results = pcb_model(img)
    annotated_img = img.copy()
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            label = f'{pcb_model.names[cls_id]} {conf:.2f}'
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            detections.append({"class": pcb_model.names[cls_id], "confidence": conf, "box": [x1, y1, x2, y2]})
    return {"image": annotated_img, "detections": detections}

# --- 4. X-Ray Bone Fracture Detection ---
FRACTURE_MODEL_PATH = 'models/bone_fracture_model.pt'
try:
    fracture_model = YOLO(FRACTURE_MODEL_PATH) if os.path.exists(FRACTURE_MODEL_PATH) else None
except Exception as e:
    print(f"加载骨折检测模型时出错: {e}")
    fracture_model = None

def detect_bone_fractures(img):
    """Detects bone fractures in an X-ray image."""
    if fracture_model is None:
        text = f"Bone Fracture Model not found at '{FRACTURE_MODEL_PATH}'"
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return {"image": img, "detections": []}
    
    results = fracture_model(img)
    annotated_img = img.copy()
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            label = f'Fracture: {conf:.2f}'
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            detections.append({"class": 'fracture', "confidence": conf, "box": [x1, y1, x2, y2]})
    return {"image": annotated_img, "detections": detections}

# ==============================================================================
#  END: CONSOLIDATED ADVANCED FUNCTIONS
# ==============================================================================

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('ascii')

def preprocess_image(img, max_size=1280):
    h, w, _ = img.shape
    if h > max_size or w > max_size:
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# --- Main Route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API Route ---
@app.route('/api/process', methods=['POST'])
def handle_processing():
    try:
        if 'image' not in request.files: return jsonify({'error': '请求中没有图像文件'}), 400
        file = request.files['image']
        operation = request.form.get('operation')
        if not operation: return jsonify({'error': '未指定操作'}), 400
        if file.filename == '' or not allowed_file(file.filename): return jsonify({'error': '无效的文件或文件类型'}), 400

        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        img = preprocess_image(img)

        params = {k: v for k, v in request.form.items() if k != 'operation'}
        for key, value in params.items():
            try:
                params[key] = float(value) if '.' in value else int(value)
            except (ValueError, TypeError): pass
        
        # --- Dispatcher ---
        processed_data = {}
        simple_ops = {
            'histogram_equalization': lambda i, **p: cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR),
            'plot_histogram': plot_histogram,
            'sharpen': sharpen,
            'add_gaussian_noise': add_gaussian_noise,
            'hough_transform': hough_transform,
            'horizontal_flip': horizontal_flip, 'vertical_flip': vertical_flip, 'cross_flip': cross_flip,
            'affine_transform': affine_transform, 'roberts_edge': roberts_edge, 'sobel_edge': sobel_edge, 'laplacian_edge': laplacian_edge
        }
        param_ops = {
            'binarize': binarize,
            'median_filter': median_filter,
            'add_salt_and_pepper_noise': add_salt_and_pepper_noise,
            'fft_lowpass': fft_lowpass,
            'fft_highpass': fft_highpass,
            'bilinear_interpolation': bilinear_interpolation, 'panning': panning, 'rotation': rotation,
            'canny_edge': canny_edge, 'erosion': erosion, 'dilation': dilation, 'open_op': open_op, 'close_op': close_op
        }

        if operation in simple_ops:
            result_img = simple_ops[operation](img)
            processed_data = {'image': image_to_base64(result_img)}
        elif operation in param_ops:
            result_img = param_ops[operation](img, **params)
            processed_data = {'image': image_to_base64(result_img)}
        elif operation.startswith('style_'):
            style_name = operation.split('_', 1)[1]
            result_img = transfer_style(img, style_name=style_name)
            processed_data = {'image': image_to_base64(result_img)}
        elif operation == 'yolov8_detect':
            result_img = detect_objects(img)
            processed_data = {'image': image_to_base64(result_img)}
        elif operation == 'pcb_defect_check':
            pcb_result = check_pcb_defects(img)
            processed_data = {'image': image_to_base64(pcb_result['image']), 'detections': pcb_result['detections']}
        elif operation == 'bone_fracture_detect':
            fracture_result = detect_bone_fractures(img)
            processed_data = {'image': image_to_base64(fracture_result['image']), 'detections': fracture_result['detections']}
        else:
            return jsonify({'error': f'未知操作: {operation}'}), 400

        return jsonify(processed_data)

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'处理时发生意外错误: {str(e)}'}), 500
