import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time
from playsound import playsound
from flask import Flask, render_template, Response

app = Flask(__name__)

mixer.init()
sound = mixer.Sound('alarm.mp3')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
model = load_model(os.path.join("models", "model.h5"))

lbl=['Close', 'Open']

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    mixer.init()
    sound = mixer.Sound('alarm.mp3')
    while(True):
        ret, frame = cap.read()
        height,width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,minNeighbors = 3,scaleFactor = 1.1,minSize=(25,25))
        eyes = eye_cascade.detectMultiScale(gray,minNeighbors = 1,scaleFactor = 1.1)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,0,0) , 3 )

        for (x,y,w,h) in eyes:

            eye = frame[y:y+h,x:x+w]
            #eye = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
            eye = cv2.resize(eye,(80,80))
            eye = eye/255
            eye = eye.reshape(80,80,3)
            eye = np.expand_dims(eye,axis=0)
            prediction = model.predict(eye)
            #print(prediction)
           #Condition for Close
            if prediction[0][0]>0.30:
                cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                score=score+1
                #print("Close Eyes")
                if(score > 15  and score<22):
                    try:
                        sound.play()
                    except:  # isplaying = False
                        pass
                elif(score>19):
                    score=16

            #Condition for Open
            elif prediction[0][1] > 0.90:
                sound.stop()
                score = score - 1
                if (score < 0):
                    score = 0
                cv2.putText(frame,"Open",(300,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                #print("Open Eyes")
                cv2.putText(frame,'Score:'+str(score),(390,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()



@app.route("/")
def home():
    
        # pass # unknown
    return render_template("index.html")

# @app.route("/start", methods=['GET', 'POST'])
# def index():
#     print(request.method)
#     if request.method == 'POST':
#         if request.form.get('Start') == 'Start':
#             return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#     else:
#         return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/test1')
def test1():
    return render_template("test1.html")


@app.route('/end')
def end():
    return render_template("end.html")

if __name__ == "__main__":
    app.run(debug=True)
