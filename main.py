#!/usr/bin/env python
# http://blog.miguelgrinberg.com/post/video-streaming-with-flask

from flask import Flask, render_template, Response, jsonify, send_from_directory, send_file
from camera import VideoCamera
import cv2
import os.path

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('templates', path)

def gen(camera, action=None):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture')
def capture():
    filename = VideoCamera().capture_to_file()
    return send_file(filename, mimetype='image/jpg')
    # cv2.imwrite("capture.jpg", VideoCamera().video.read())
    #return Response(VideoCamera().get_frame(), mimetype='image/jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=33868, debug=False)
