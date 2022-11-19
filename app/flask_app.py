import os
import cv2
import numpy as np
from flask import Flask, render_template, Response
import tensorflow as tf
from gtts import gTTS
import time

global graph
global writer
from skimage.transform import resize
from keras.models import load_model

graph = tf.compat.v1.get_default_graph()
writer = None

model = load_model("aslpng.h5")

vals = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

app = Flask(__name__)

print("[INFO] Accessing Video Stream...")

vs = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = vs.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


pred = ""


@app.route("/")
def index():
    return render_template("index.html")


def detect(frame):
    img = resize(frame, (64, 64, 1))
    img = np.expand_dims(img, axis=0)
    if np.max(img) > 1:
        img = img / 255.0
    with graph.as_default():
        prediction = model.predict_classes(img)
    print(prediction)
    pred = vals[prediction[0]]
    print(pred)
    return pred


def gen_prediction_text():
    while True:
        success, frame = vs.read()  # read the camera frame
        if not success:
            break
        else:
            prediction = detect(frame)
        yield prediction
        time.sleep(0.2)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/text_feed")
def text_feed():
    return Response(gen_prediction_text(), mimetype="text")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
