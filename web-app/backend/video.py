import cv2
from datetime import datetime
import numpy as np
import os
import pathlib

import tensorflow as tf
from backend.models import CNNModel,ResBlock
from imutils.object_detection import non_max_suppression
# Camera

camera=cv2.VideoCapture(0)

# Loading Models

# Emotion Model
emotion_model = tf.keras.models.load_model('backend/models/vgg.h5')

# Face Model
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
face_model = cv2.CascadeClassifier(str(cascade_path))

# categories
categories = ['angry','disgust','fear','happy','neutral','sad','surprise']

class Capture():
    def __init__(self):
        self.video=cv2.VideoCapture(0)
        self.path=os.path.dirname(os.path.realpath(__file__))
        self.path=self.path.replace('\\','/')
        
        self.face_model = face_model
        self.emotion_model = emotion_model
        
    def __del__(self):
        self.video.release()
    
    def show_video(self,save=None):
        # for computation handling
        
        ret,frame= self.video.read()
        frame=cv2.resize(frame,(640,480))
        frame=cv2.flip(frame,180)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        # for Face detections
        faces = self.face_model.detectMultiScale(gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
            )
        
        faces = non_max_suppression(faces)
    
        for (x, y, width, height ) in faces:
            cv2.rectangle(frame, (x,y), (x + width, y + height ), (0,255,0), 2)

            # Detecting emotions
            x_min,y_min,x_max,y_max = x , y, x + width, y + height
            emotion = self.predict(frame[y_min:y_max , x_min:x_max])
            # emoji_image = cv2.imread(f'backend/emojis/{emotion}.png', cv2.IMREAD_UNCHANGED)
            # emoji_image = emoji_image[:, :, :3]
            font_scale = 1
            font_thickness = 2
            font_color = (0, 0, 255)  # Red color
            font = cv2.FONT_HERSHEY_SIMPLEX


            # Get the size of the text to determine its position
            (text_width, text_height), _ = cv2.getTextSize(f'{emotion}', font, font_scale, font_thickness)
            text_x = x + int((width - text_width) / 2)
            text_y = y + height + 30    
            cv2.putText(frame, f'{emotion}', (text_x, text_y), font, font_scale, font_color, font_thickness)

            # frame[y:y + emoji_height, x:x + emoji_width] = emoji_img_resized
        
            
        if save is not None:
            now_time=datetime.now()
            current_time = now_time.strftime("%d_%m_%Y_%H_%M_%S")
            cv2.imwrite(f'saved_snaps/{current_time}.jpg',frame)
            save=None
            
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()
    
    def predict(self,image):

        # Resize the image to the desired size (48x48 pixels with 3 channels)
        desired_size = (48, 48)
        resized_img = cv2.resize(image, desired_size)

        prediction = self.emotion_model.predict(resized_img.reshape(1, 48, 48, 3))

        final_prediction = np.argmax(prediction)

        return(categories[final_prediction].upper())
        

def generate_video(camera,save=None):
    while True:
        frame=camera.show_video(save=save)
        
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

def save_image():
    pass