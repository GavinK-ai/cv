import os
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for
from PIL import Image
import io
import base64
import mimetypes
from flask import Flask, request, make_response, jsonify, render_template, redirect, url_for
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


# @app.route('/api/cv/upload', methods=['GET', 'POST', 'PUT', 'DELETE'])
# def upload():
#     title = 'Please upload your image'
    
#     name = ''

#     if request.method == 'POST':
#         message=''
#         title += ': POST'
#         name = request.form['name']
#         message += name
#         pix = request.form['pix']
#         message += pix

#         pixFile = request.files['pix']
#     elif request.method == 'PUT':
#         title += ': PUT'
#     elif request.method == 'DELETE':
#         title += ': DELETE'
#     else:
#         title += ': GET'

#     return render_template('upload.html', title = title, message = message)

UPLOAD_FOLDER = r'C:\SDK\Perantis\Perantis\cv\gavPrj2\static\uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
save_mode = 0  # 1001

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/cv/upload', methods=['GET', 'POST'])
def upload_file():
    # get querystring
    filename = '' if request.args.get(
        'filename') is None else request.args.get('filename')
    uri = '' if request.args.get('uri') is None else request.args.get('uri')
    uri2 = '' if request.args.get('uri2') is None else request.args.get('uri2')
    #
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            root, ext = os.path.splitext(filename)
            print(root, ext)
            ext = ext.lower()
            ext = '.jpeg' if ext == '.jpg' else ext

            if save_mode == 0:
                file.save(filePath)
                uri = f'/static/uploads/{filename}'
            else:
                f = file.read()
                print('file-len', len(f))
                imgArray = np.frombuffer(f, np.uint8)

                # create image
                img = cv.imdecode(imgArray, cv.IMREAD_COLOR)

                if save_mode == 1000:
                    # write image to path
                    cv.imwrite(filePath, img)
                    uri = f'/static/uploads/{filename}'

                mime = mimetypes.types_map[ext]
                if save_mode == 1010:
                    # transform to base64 url
                    # 1
                    uri = to_base64_uri_pil(img, ext, mime)

                if save_mode == 1001:
                    # 2
                    uri = to_base64_uri(img, ext, mime)

            return redirect(url_for('upload_file', uri=uri))

    return f'''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>      
    </form>
    <img src="{uri}" />
    '''

def to_base64_uri_pil(img, ext, mime):
    imgRGB = img[:, :, ::-1]
    imgPIL = Image.fromarray(imgRGB)
    buff = io.BytesIO()

    imgFormat = ext[1:]
    print(imgFormat)

    imgPIL.save(buff, format=imgFormat)
    imgBase64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    uri = f"data:{mime};base64,{imgBase64}"
    return uri


def to_base64_uri(img, ext, mime):
    retval, buffer = cv.imencode(ext, img)
    imgBase64_2 = base64.b64encode(buffer).decode("utf-8")

    uri2 = f"data:{mime};base64,{imgBase64_2}"
    return uri2

if __name__ == "__main__":
    app.run(host='0.0.0.0')