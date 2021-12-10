import os
from flask import Flask, request, url_for
from flask.helpers import flash
from werkzeug.utils import redirect, secure_filename
import cv2 as cv
import numpy as np
import tensorflow as tf

UPLOAD_FOLDER = r'C:\SDK\Perantis\Perantis\cv\gavPrj2\static\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
save_mode = 0  # 1001

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#/ 
@app.route('/face', methods=['GET', 'POST'])
def face():

    # /?originalUri=xxxx&resizedUri=yyyy
    originalUri = '' if request.args.get('orignalUri') is None else request.args.get('orignalUri')
    resizedUri = '' if request.args.get('resizedUri') is None else request.args.get('resizedUri')
    error = '' if request.args.get('error') is None else request.args.get('error')

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            error = 'No file part'
            return redirect(url_for('index', error = error))

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            error = 'No selected file'
            return redirect(url_for('index', error = error))

        # read the uploaded image
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # fix file extension
            root, ext = os.path.splitext(filename)
            print(root, ext)
            ext = ext.lower()
            ext = '.jpeg' if ext == '.jpg' else ext

            # get raw file
            f = file.read()
            
            # convert to numpy array 1D
            imgArray = np.frombuffer(f, np.uint8)

            # create image by converting the 1D array
            img = cv.imdecode(imgArray, cv.IMREAD_COLOR)

            # get file path to save
            
            originalFileName = 'original' + ext
            originalUri = save_file(img,originalFileName)
            #filePath = os.path.join(app.config['UPLOAD_FOLDER'], originalFileName)

            # save the original image
            #cv.imwrite(filePath, img)

            originalUri = originalFileName
            # resize the image
            imgResized = cv.resize(img,(200,200), interpolation= cv.INTER_NEAREST)
            imgResized[:,:,0] = 0

            resizedFileName = 'resized' + ext
            resizedlUri = save_file(imgResized,resizedFileName)
            resizedUri = resizedFileName

            testImages = np.reshape(imgResized,(1,200,200,3))

            exportPath = '../tf_model4/4_max'
            newModel = tf.keras.models.load_model(exportPath)
            probabilityModel = tf.keras.Sequential([newModel, tf.keras.layers.Softmax()])
            newPredictions = probabilityModel.predict(testImages)

            i = np.argmax(newPredictions[0])
            predictLabel = classNames[i]

        return redirect(url_for('index',originalUri=originalUri, resizedUri=resizedUri))

            
        # redirect to GET
    
    #print(originalUri,resizedUri,error)
    return f'''

    <!doctype html>
    <title>Computer Vision</title>

    <h1>Upload Image</h1>

    <div style = "color:red">{error}</div>

    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>      
    </form>
    
    <div>
        <h3>Original</h3>
        <p>{originalUri}</p>
        <img src="static/uploads/{originalUri}" />
    </div>

    <div>
        <h3>Resized</h3>
        <p>{resizedUri}</p>
        <img src="static/uploads/{resizedUri}" />
    </div>

    '''
    



if __name__ == "__main__":
    app.run(host="0.0.0.0")