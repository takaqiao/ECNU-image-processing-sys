import os
from flask import Flask

# 1. 在这里创建并配置核心的 Flask app 对象
app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)

# 2. 在 app 对象上设置配置
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 3. 在文件的最后导入路由模块(main.py)
# 这是为了确保在导入路由时，app 对象已经完全创建好了
from app import main
