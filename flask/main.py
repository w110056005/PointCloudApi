import subprocess
import datetime
from flask import Flask, request, send_file
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
    for file in files:
      print(file.name)
      file.save(file.filename)

    output_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+ext

    print(output_name)
    # p = subprocess.run(
    #     [
    #         'python', 'global_registration.py',
    #         files[0].filename,
    #         files[1].filename,
    #         output_name
    #     ]
    # )
    return output_name


if __name__ == "__main__":
    app.run(debug=True)
