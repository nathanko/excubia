# camera.py

import cv2

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib

# http://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
#args = vars(ap.parse_args())
#print (args)
#print ("----------")
args = {'shape_predictor': 'data/shape_predictor_68_face_landmarks.dat'}

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        #print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(args["shape_predictor"])

        success, image = self.video.read()
        
        # https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/
        cascPath = "data/haarcascade_frontalface_default.xml"
        #cascEyePath = "data/haarcascade_eye.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        #eyeCascade = cv2.CascadeClassifier(cascEyePath)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then convert
            # facial landmark coordinates to a Numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # loop over the (x, y)-coordinates for the facial landmark and draw
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(15, 15),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Eyes detection
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = image[y:y+h, x:x+w]
            #eyes = eyeCascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh),(255, 0, 255), 2)
        #print ("Number of faces: {}".format(len(faces)))


        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()
