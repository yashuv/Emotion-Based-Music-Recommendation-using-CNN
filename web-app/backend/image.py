import cv2
import numpy as np
import pandas as pd
import pathlib

import tensorflow as tf
import random
from backend.models import CNNModel,ResBlock


# Loading Models

# Emotion Model

emotion_model = tf.keras.models.load_model('backend/models/vgg.h5')

# Face Model

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
face_model = cv2.CascadeClassifier(str(cascade_path))

# categories

categories = ['angry','disgust','fear','happy','neutral','sad','surprise']

# music
df_music = pd.read_csv('backend/dataframes/music.csv')


class ImageAndMusic():
    
    def __init__(self,image):
        self.face_model = face_model
        self.emotion_model = emotion_model
        self.image = image
    
    # predict image

    def predict_image_and_recommend(self):

        gray=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
            )
        if len(faces) != 0 :
            for (x, y, width, height ) in faces:
                x_min,y_min,x_max,y_max = x , y, x + width, y + height
                prediction = self.predict(self.image[y_min:y_max , x_min:x_max])
        else:
            # Or Predict Image
            prediction = self.predict(self.image)
        
        
        recommended_song = self.recommend(prediction) 
        return prediction,recommended_song
    
    # Predict image
    
    def predict(self,image):
        # Resize the image to the desired size (48x48 pixels with 3 channels)
        
        desired_size = (48, 48)
        resized_img = cv2.resize(image, desired_size)

        prediction = emotion_model.predict(resized_img.reshape(1, 48, 48, 3))
        # prediction = emotion_model.predict(resized_img)


        final_prediction = np.argmax(prediction)
        return(categories[final_prediction].upper())
    
    def recommend(self,category):
        
        if category in ['fear','angry','surprise']:
            df_filtered = df_music[df_music["mood"] == 'Energetic'].reset_index(drop=True)
            recommended_song = random.choice(df_filtered['name'])
        elif category == 'happy':
            df_filtered = df_music[df_music["mood"] == 'Happy'].reset_index(drop=True)
            recommended_song = random.choice(df_filtered['name'])
        elif category == 'sad':
            df_filtered = df_music[df_music["mood"] == 'Sad'].reset_index(drop=True)
            recommended_song = random.choice(df_filtered['name'])
        else:
            df_filtered = df_music[df_music["mood"] == 'Calm'].reset_index(drop=True)
            recommended_song = random.choice(df_filtered['name'])

        return recommended_song
        