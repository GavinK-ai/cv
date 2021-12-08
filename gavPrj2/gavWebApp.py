from PIL import Image
from flask import Flask, request, make_response
import urllib.request
import numpy as np
import cv2 as cv

app = Flask(__name__)

@app.route("/")
def index():
    return "Welcome to Index Page"

@app.route("/hello")
def hello_world():
    return "<h1>Hello World!</h1>"

@app.route("/user/<username>")
def hello_user(username):
    return f"Welcome, {username}!"

@app.route("/displayimg/gavin")
def displayimg():
    return f"<img src='gavin_photo.png' width='42' height='42'></img>"


@app.route("/add/<a>/<b>")
def add(a,b):
    return f"{int(a)}+{int(b)}={int(a)+int(b)}"
@app.route("/subtract/<a>/<b>")
def subtract(a,b):
    return f"{int(a)}-{int(b)}={int(a)-int(b)}"
@app.route("/multiply/<a>/<b>")
def multiply(a,b):
    return f"{int(a)}*{int(b)}={int(a)*int(b)}"
@app.route("/divide/<a>/<b>")
def divide(a,b):
    return f"{int(a)}/{int(b)}={float(int(a)/int(b))}"
@app.route("/remainder/<a>/<b>")
def remainder(a,b):
    return f"{int(a)}%{int(b)}={float(int(a)%int(b))}"
def hello_user(username):
    return f"Welcome, {username}!"


@app.route("/add2",methods=['GET'])
def add2():
    
    
    a = request.args.get('a')
    b = request.args.get('b')

    return f"{int(a)}+{int(b)}={int(a)+int(b)}"

@app.route('/display/', methods=['GET'])
def display():
    print('display')
    externalURL = r'https://raw.githubusercontent.com/goke-ai/cv/master/essential/assets/Picture1.png'
    fileURL = r'file:///SDK/Perantis/Perantis/cv/samples/data/face.jpg'
    
    imgUrl = request.args.get('url')
    
    if imgUrl is None:
        imgUrl = fileURL
        
    # Get the image:
    with urllib.request.urlopen(imgUrl) as url:
        img = np.asarray(bytearray(url.read()), dtype=np.uint8)

    print(img.shape)
    
    # Convert the image to OpenCV format:
    imgBGR = cv.imdecode(img, -1)
    
    print(imgBGR.shape)

    
    # Compress the image and store it in the memory buffer:
    retval, buffer = cv.imencode('.jpeg', imgBGR)
    
    print(len(buffer))
    print(buffer.shape)
    
    # Build the response:
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    # Return the response:

    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0')