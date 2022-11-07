from flask import Flask, request, send_file
from flasgger import Swagger
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash

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

if __name__ == "__main__":
    app.run(debug=True)
