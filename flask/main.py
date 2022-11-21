import subprocess
import datetime
import os
import traceback

from flask import Flask, request, send_from_directory, abort
from flasgger import Swagger
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash
from pathlib import Path

def abort_msg(e):
    """500 bad request for exception

    Returns:
        500 and msg which caused problems
    """
    error_class = e.__class__.__name__  # 引發錯誤的 class
    detail = e.args[0]  # 得到詳細的訊息
    cl, exc, tb = sys.exc_info()  # 得到錯誤的完整資訊 Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1]  # 取得最後一行的錯誤訊息
    fileName = lastCallStack[0]  # 錯誤的檔案位置名稱
    lineNum = lastCallStack[1]  # 錯誤行數
    funcName = lastCallStack[2]  # function 名稱
    # generate the error message
    errMsg = "Exception raise in file: {}, line {}, in {}: [{}] {}.".format(fileName, lineNum, funcName, error_class, detail)
    # return 500 code
    abort(500, errMsg)

app = Flask(__name__)
app.config['SWAGGER'] = {
    "title": "Flask API",
    "description": "SWAGGER API",
    "version": "1.0.0",
    "termsOfService": "",
    "hide_top_bar": True
}
# Auth Setting
auth = HTTPBasicAuth()
user = 'awinlab'
pw = 'awinlab'
users = {
    user: generate_password_hash(pw)
}

swagger_template = {
    'securityDefinitions': {
        'basicAuth': {
            'type': 'basic'
        }
    },
}


Swagger(app, template=swagger_template)

DOWNLOAD_DIRECTORY = '/app/files/'


@auth.verify_password
def verify_password(username, password):
    if username in users:
        return check_password_hash(users.get(username), password)
    return False

@app.route('/files/<id>', methods=['GET'])
@auth.login_required
def get_files():
    """

      Get registrated point cloud file
      ---
      tags:
        - Node APIs
      produces: application/json,
      parameters:
      - name: id
        in: path
        type: string
        required: true
      responses:
        200:
          description: Return pcd
    """
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY+id, id+'.pcd', as_attachment=True)
    except Exception as e:
        abort_msg(e)

@app.route('/file', methods=['GET'])
@auth.login_required
def get_file():
    """
      Get file
      ---
      tags:
        - Node APIs
      produces: application/json,
      responses:
        200:
          description: Return pcd
    """
    try:
        return send_from_directory('/app', 'test.txt', as_attachment=True)
    except Exception as e:
        abort_msg(e)

@app.route('/registration', methods=['POST'])
@auth.login_required
def registration():
    """
      Merge two Point Cloud files and retrun the result file name
      ---
      tags:
        - Node APIs
      produces: application/json,
      responses:
        200:
          description: Merge two Point Cloud files 
          examples:
            "20221107210100147.pcd"
    """
    files = request.files.getlist("file")
    ext = Path(files[0].filename).suffix

    prefix = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    output_name = prefix+ext
    file_folder = DOWNLOAD_DIRECTORY+prefix+'/'
    os.makedirs(file_folder)

    for file in files:
        file.save(file_folder+file.filename)

    p = subprocess.run(
        [
            'python', 'global_registration.py',
            file_folder+files[0].filename,
            file_folder+files[1].filename,
            file_folder+output_name
        ]
    )
    return output_name


if __name__ == "__main__":
    app.run(debug=True)
