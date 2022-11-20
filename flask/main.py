import subprocess
import datetime
import os
from flask import Flask, request, send_from_directory
from flasgger import Swagger
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash
from pathlib import Path


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


@app.route('/hello-world', methods=['GET'])
@auth.login_required
def hello_world():
    """
      Return Hello World 
      ---
      tags:
        - Node APIs
      produces: application/json,
      responses:
        200:
          description: Return Hello World 
          examples:
            "Hello World"
    """
    return "Hello World"


@app.route('/files/<id>', methods=['GET'])
@auth.login_required
def get_files():
   """
     Get registrated point cloud file
     ---
     tags:
       - Node APIs
     produces: application/json,
     responses:
       200:
         description: Return pcd
   """
   try:
       return send_from_directory(DOWNLOAD_DIRECTORY, id+'.pcd', as_attachment=True)
   except FileNotFoundError:
       abort(404)


# @app.route('/registration', methods=['POST'])
# @auth.login_required
# def registration():
#     """
#       Merge two Point Cloud files and retrun the result file name
#       ---
#       tags:
#         - Node APIs
#       produces: application/json,
#       responses:
#         200:
#           description: Merge two Point Cloud files 
#           examples:
#             "20221107210100147.pcd"
#     """
#     files = request.files.getlist("file")
#     ext = Path(files[0].filename).suffix

#     prefix = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
#     output_name = prefix+ext
#     file_folder = DOWNLOAD_DIRECTORY+prefix+'/'
#     os.makedirs(file_folder)

#     for file in files:
#         file.save(file_folder+file.filename)

#     p = subprocess.run(
#         [
#             'python', 'global_registration.py',
#             file_folder+files[0].filename,
#             file_folder+files[1].filename,
#             file_folder+output_name
#         ]
#     )
#     return output_name


if __name__ == "__main__":
    app.run(debug=True)
