from PIL import Image
from flask import Flask, request, make_response, jsonify, render_template
import numpy as np
import cv2 as cv

app = Flask(__name__)


@app.errorhandler(500)
def not_found(e):
    return jsonify({
        "status": "internal error",
        "massage": "internal error occurred in server"
    }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "not found", "massage": "route not found"}), 404


# @app.errorhandler(400)
# def not_found(e):
#     return jsonify({"status":"not ok","massage": "this server could not understand your request"}),400

#data = ['afiq', 'azureen', 'gavin', 'goke', 'inamul', 'jincheng', 'mahmuda', 'numan', 'saseendran']
data = {
    '1': 'afiq',
    '2': 'azureen',
    '3': 'gavin',
    '4': 'goke',
    '5': 'inamul',
    '6': 'jincheng',
    '7': 'mahmuda',
    '8': 'numan',
    '9': 'saseendran'
}


@app.route('/')
def index():
    page = 'index'
    description = """This is an api for computer vision class. The api gets you the list of the student in class.
    Please enjoy and have some fun.
    the api routes are the following:
    1. get all: api/cv
    2. get by id: api/cv/5
    3. post: api/cv
    4. put by id: api/cv/5
    5. delete by id: api/cv/5

    Thanks for check our api out
    """
    return render_template('index.html', page = page, description = description, data = data)


@app.route('/api/cv', methods=['GET'])
def get_all():
    return jsonify({"status": "ok", "student": data}), 200


@app.route('/api/cv/<int:id>', methods=['GET'])
def get_by_id(id):

    if id == 0:
        return jsonify({
            "status":
            "not ok",
            "massage":
            "this server could not understand your request"
        }), 400

    student = data[str(id)]

    return jsonify({"status": "ok", "students": student}), 200


@app.route('/api/cv', methods=['POST'])
def post():
    data_list = [int(i) for i in data.keys()]
    max_id = max(data_list)
    name = request.json['name']
    id = max_id+1
    key = str(id)
    data[key]=name

    return jsonify({"status": "ok", "id": len(data),"name": name}), 200


@app.route('/api/cv/<int:id>', methods=['PUT'])
def put(id):

    if id < 1:
        return jsonify({
            "status":
            "not ok",
            "massage":
            "this server could not understand your request"
        }), 400
    
    name = request.json['name']
    data[str(id)]=name

    return jsonify({
        "status": "ok",
        "student": {
            "id": id,
            'name': name
        }
    }), 200


@app.route('/api/cv/<int:id>', methods=['DELETE'])
def delete(id):

    if id == 0:
        return jsonify({
            "status":
            "not ok",
            "massage":
            "this server could not understand your request"
        }), 400

    del data[str(id)]

    return jsonify({"status": "ok"}), 200


@app.route('/api/cv/all<int:id>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def all(id):
    message = 'GET'
    status = 'ok'
    bodyJson = request.json

    if request.method == 'POST':
        message = 'POST'
    elif request.method == 'PUT':
        message = 'PUT'
    elif request.method == 'DELETE':
        message = 'DELETE'
    else:
        message = 'GET'

    return jsonify({
        "status": status,
        "message": message,
        "bodyJson": bodyJson,
        "id": id
    }), 200


@app.route('/api/cv/upload', methods=['GET', 'POST', 'PUT', 'DELETE'])
def upload():
    title = 'Please upload your image'
    
    name = ''

    if request.method == 'POST':
        message=''
        title += ': POST'
        name = request.form['name']
        message += name
        pix = request.form['pix']
        message += pix

        pixFile = request.files['pix']
    elif request.method == 'PUT':
        title += ': PUT'
    elif request.method == 'DELETE':
        title += ': DELETE'
    else:
        title += ': GET'

    return render_template('upload.html', title = title, message = message)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0')