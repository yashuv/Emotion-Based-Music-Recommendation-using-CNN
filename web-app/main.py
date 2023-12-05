from flask import Flask,render_template,Response,request,redirect

from backend.video import *
from backend.image import * 
from backend.pre_process import * 

import cv2,os,atexit


app=Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Landing Page

@app.route('/', methods=['GET', 'POST'])
def index():
    image_path = 'static/images/uploaded.jpg'
    
    if request.method == "POST":
        button_value = request.form.get("button-value")

        if button_value == 'live':
            generate_video(Capture(),save=None)
            return render_template('video_detection.html', mimetype='multipart/x-mixed-replace;boundary=frame')
        elif button_value == 'image':
            placeholder = cv2.imread('static/images/placeholder.png')
            cv2.imwrite(image_path,placeholder)
            return render_template('image_detection.html')
    return render_template('landing.html')

# For Video

@app.route('/live/')
def video():
    return Response(generate_video(Capture(),save=None),mimetype='multipart/x-mixed-replace;boundary=frame')

# Image detection

@app.route('/image/', methods=['GET','POST'])
def image_detection():
    
    if request.method == 'POST':
        
        image_path = 'static/images/uploaded.jpg'
        image = request.files['image']        

        if image:
            image_data = image.read()
            image_numpy = image_to_numpy(image_data)
            prediction,recommended_song = ImageAndMusic(image_numpy).predict_image_and_recommend()
            cv2.imwrite(image_path,image_numpy)
            return render_template('image_detection.html',prediction=prediction,recommended_song = recommended_song)
        
    return render_template('image_detection.html')



# Snapshot of video

@app.route('/snapshot/')
def Snapshot(): 
    #To Capture
    Capture().show_video(save=1)
   
    return render_template('video_detection.html', mimetype='multipart/x-mixed-replace;boundary=frame')

# On exit delete all filter

def OnExitApp():
    try:
        os.remove('static/images/uploaded.jpg')
    except:
        pass

atexit.register(OnExitApp)


if __name__=="__main__":
    
    app.run(host='0.0.0.0',port=5050,debug=True)