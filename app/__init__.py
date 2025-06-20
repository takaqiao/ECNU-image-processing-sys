import os
from flask import Flask

# 初始化 Flask 应用实例
# 指定了模板和静态文件的相对路径，以确保应用能正确找到它们
app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)

# 应用配置
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 在配置完成后导入路由模块，以避免循环导入问题
from app import main