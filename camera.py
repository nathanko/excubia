# camera.py

import cv2
import random
import os
from imutils import face_utils
import numpy as np
import imutils
# import dlib
# import pymongo
# from pymongo import MongoClient

# http://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
#https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

args = {'cascPath': 'data/haarcascade_frontalface_default.xml',
        'shape_predictor': 'data/shape_predictor_68_face_landmarks.dat'}
# client = MongoClient('mongodb://excubia:Incredible54-@ds139884.mlab.com:39884/excubia-faces')
# db = client['excubia-faces']
# print(db)
# collection = db.alpha
# print(collection)



class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        #print("[INFO] loading facial landmark predictor...")

        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor(args["shape_predictor"])

        success, image = self.video.read()
        
        # https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/
        #cascEyePath = "data/haarcascade_eye.xml"
        faceCascade = cv2.CascadeClassifier(args["cascPath"])
        #eyeCascade = cv2.CascadeClassifier(cascEyePath)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # detect faces in the grayscale frame
        # rects = detector(gray, 0)
        # faceLandmarks = []

        # # loop over the face detections
        # for rect in rects:
        #     # determine the facial landmarks for the face region, then convert
        #     # facial landmark coordinates to a Numpy array
        #     shape = predictor(gray, rect)
        #     shape = face_utils.shape_to_np(shape)
        #     # loop over the (x, y)-coordinates for the facial landmark and draw
        #     for (x, y) in shape:
        #         cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        #         #print("({},{})".format(x,y))
        #         faceLandmarks.append((x,y))
        
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
            # post = {"landmarks": []}
            # for (s,t) in faceLandmarks:
            #     if ( x <= s <= x+s and y <= t <= y+h):
            #         # Add to database
            #         post['landmarks'].append([str(s),str(t)])
            # try:
            #     post_id = db.posts.insert_one(post).inserted_id
            # except Exception as e:
            #     pass
            # Eyes detection
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = image[y:y+h, x:x+w]
            #eyes = eyeCascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh),(255, 0, 255), 2)
        #print ("Number of faces: {}".format(len(faces)))

        
        cv2.putText(image,"Current # of people: "+str(len(faces)),(9,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        ret, jpeg = cv2.imencode('.jpg', image)
        
        #Write to file to temp00.jpg
        cv2.imwrite("temp00.jpg", image)
        #Move the file to temp.jpg when done, ensures temp.jpg is always a complete file
        os.replace("temp00.jpg", "temp.jpg")
        
        return jpeg.tobytes()

    def capture_to_file(self):
        os.replace("temp.jpg", "capture.jpg")
        print("captured")
        return "capture.jpg"
