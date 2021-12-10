import os
import cv2 as cv
from flask import Flask, request, flash, redirect, url_for
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf


UPLOAD_FOLDER = r'C:\SDK\Perantis\Perantis\cv\gavPrj2\static\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

save_mode = 0  # 1001


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# /


@app.route('/', methods=['GET', 'POST'])
def index():

    # /?originalUri=xxxxxx&resizedUri=yyyyyyy
    originalUri = '' if request.args.get(
        'originalUri') is None else request.args.get('originalUri')
    resizedUri = '' if request.args.get(
        'resizedUri') is None else request.args.get('resizedUri')
    error = '' if request.args.get(
        'error') is None else request.args.get('error')

    if request.method == 'POST':
        # read the uploaded image
        # check if the post request has the file part
        if 'imgFile' not in request.files:
            error = 'No file part'
            return redirect(url_for('index', error=error))

        file = request.files['imgFile']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return redirect(url_for('index', error=error))

        # check if allowed file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # fix file extension
            root, ext = os.path.splitext(filename)
            ext = ext.lower()
            ext = '.jpeg' if ext == '.jpg' else ext

            # get raw file
            f = file.read()

            # convert to numpy array 1D
            imgArray = np.frombuffer(f, np.uint8)

            # create image by converting the 1D array
            img = cv.imdecode(imgArray, cv.IMREAD_COLOR)

            # save the original image
            # get file path to save
            originalFileName = 'original' + ext
            originalUri = save_file(img, originalFileName)

        # resize the image
            imgResized = cv.resize(
                img, (200, 200), interpolation=cv.INTER_NEAREST)
            imgResized[:, :, 0] = 0
            # save the resized image
            # get file path to save
            resizedFileName = 'resize' + ext
            resizedUri = save_file(imgResized, resizedFileName)

        # redirect to GET and display image
        return redirect(url_for('index', originalUri=originalUri, resizedUri=resizedUri))

    return f'''
    <!doctype html>
    <title>Computer Vision</title>
    <h1>Upload Image</h1>
    <div style="color: red">
      {error}
    </div>
    <form method=post enctype=multipart/form-data>
      <input type=file name=imgFile>
      <input type=submit value=Upload>
    </form>
    <div>
      <h3>Original</h3>
      <div>{originalUri}</div>
      <img src="static/uploads/{originalUri}" />
    </div>
    <div>
      <h3>Resized</h3>
      <img src="static/uploads/{resizedUri}" />
    </div>
    '''


def save_file(img, originalFileName):
    filePath = os.path.join(app.config['UPLOAD_FOLDER'], originalFileName)
    cv.imwrite(filePath, img)
    originalUri = originalFileName
    return originalUri


# /
@app.route('/face', methods=['GET', 'POST'])
def face():

    # /?originalUri=xxxxxx&resizedUri=yyyyyyy
    originalUri = '' if request.args.get(
        'originalUri') is None else request.args.get('originalUri')
    predictedLabel = '' if request.args.get(
        'predictedLabel') is None else request.args.get('predictedLabel')
    error = '' if request.args.get(
        'error') is None else request.args.get('error')

    if request.method == 'POST':
        # read the uploaded image
        # check if the post request has the file part
        if 'imgFile' not in request.files:
            error = 'No file part'
            return redirect(url_for('face', error=error))

        file = request.files['imgFile']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return redirect(url_for('face', error=error))

        # check if allowed file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # fix file extension
            root, ext = os.path.splitext(filename)
            ext = ext.lower()
            ext = '.jpeg' if ext == '.jpg' else ext

            # get raw file
            f = file.read()

            # convert to numpy array 1D
            imgArray = np.frombuffer(f, np.uint8)

            # create image by converting the 1D array
            img = cv.imdecode(imgArray, cv.IMREAD_COLOR)

            # save the original image
            # get file path to save
            originalFileName = 'original' + ext
            originalUri = save_file(img, originalFileName)

            # prediction
            classNames = ['afiq',
                            'azureen',
                            'gavin',
                            'goke',
                            'inamul',
                            'jincheng',
                            'mahmuda',
                            'numan',
                            'saseendran']
            # prepare test images
            # resize the image
            imgResized = cv.resize(
                img, (200, 200), interpolation=cv.INTER_NEAREST)
            testImages = np.reshape(imgResized, (1, 200, 200, 3))

            # load SavedModel format
            exportPath = 'C:/SDK/Perantis/Perantis/cv/gavPrj/tf_model3/4_max'
            model = tf.keras.models.load_model(exportPath)

            # predict
            probabilityModel = tf.keras.Sequential(
                [model, tf.keras.layers.Softmax()])

            predictions = probabilityModel.predict(testImages)

            i = np.argmax(predictions[0])
            predictedLabel = classNames[i]      
            
        # redirect to GET and display image
        return redirect(url_for('face', originalUri=originalUri, predictedLabel=predictedLabel))

    return f'''
    <!doctype html>
    <title>Computer Vision</title>
    <h1>Upload Face Photo</h1>
    <div style="color: red">
      {error}
    </div>
    <form method=post enctype=multipart/form-data>
      <input type=file name=imgFile>
      <input type=submit value=Upload>
    </form>
    <div>
      <h3>Prediction</h3>
      <p style="color: blue">{predictedLabel}<p>
      <img src="static/uploads/{originalUri}" />
    </div>
    '''

# /
@app.route('/tire', methods=['GET', 'POST'])
def tire():

    # /?originalUri=xxxxxx&resizedUri=yyyyyyy
    originalUri = '' if request.args.get(
        'originalUri') is None else request.args.get('originalUri')
    predictedLabel = '' if request.args.get(
        'predictedLabel') is None else request.args.get('predictedLabel')
    error = '' if request.args.get(
        'error') is None else request.args.get('error')

    if request.method == 'POST':
        # read the uploaded image
        # check if the post request has the file part
        if 'imgFile' not in request.files:
            error = 'No file part'
            return redirect(url_for('tire', error=error))

        file = request.files['imgFile']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return redirect(url_for('tire', error=error))

        # check if allowed file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # fix file extension
            root, ext = os.path.splitext(filename)
            ext = ext.lower()
            ext = '.jpeg' if ext == '.jpg' else ext

            # get raw file
            f = file.read()

            # convert to numpy array 1D
            imgArray = np.frombuffer(f, np.uint8)

            # create image by converting the 1D array
            img = cv.imdecode(imgArray, cv.IMREAD_COLOR)

            # save the original image
            # get file path to save
            originalFileName = 'original' + ext
            originalUri = save_file(img, originalFileName)

            # prediction
            classNames = {0:'cracked', 1:'normal'}
            # prepare test images
            # resize the image
            imgResized = cv.resize(
                img, (224, 224), interpolation=cv.INTER_NEAREST)
            testImages = np.reshape(imgResized, (1, 224, 224, 3))

            # load SavedModel format
            exportPath = 'cv_image_resnet_85'
            model = tf.keras.models.load_model(exportPath)
            #print(model.summary())
            # predict
            probabilityModel = tf.keras.Sequential(
                [model, tf.keras.layers.Softmax()])

            predictions = probabilityModel.predict(testImages)

            i = np.argmax(predictions[0])
            predictedLabel = classNames[i]      
            
        # redirect to GET and display image
        return redirect(url_for('tire', originalUri=originalUri, predictedLabel=predictedLabel))

    return f'''
    <!doctype html>
    <title>Computer Vision</title>
    <h1>Upload Tire Photo</h1>
    <div style="color: red">
      {error}
    </div>
    <form method=post enctype=multipart/form-data>
      <input type=file name=imgFile>
      <input type=submit value=Upload>
    </form>
    <div>
      <h3>Prediction</h3>
      <p style="color: blue">{predictedLabel}<p>
      <img src="static/uploads/{originalUri}" />
    </div>
    '''
if __name__ == "__main__":
    app.run(host= "0.0.0.0")
