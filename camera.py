# camera.py

import cv2

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(1)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        
        # https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/

        cascPath = "haarcascades/haarcascade_frontalface_default.xml"
        cascEyePath = "haarcascades/haarcascade_eye.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        eyeCascade = cv2.CascadeClassifier(cascEyePath)

        #Cool AI Stuff
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]

            eyes = eyeCascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh),(255, 0, 255), 2)
        print ("Number of faces: {}".format(len(faces)))


        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()
