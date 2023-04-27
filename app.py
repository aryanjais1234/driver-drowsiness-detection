from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import dlib
from imutils import face_utils
from playsound import playsound
import os
app = Flask(__name__)


def gen_frames():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    sleepy = 0
    drowsy = 0
    active = 0
    status = ""
    color = (0, 0, 0)

    def compute(ptA, ptB):
        distance = np.linalg.norm(ptA-ptB)
        return distance

    def blinked(a, b, c, d, e, f):
        up = compute(b, d) + compute(c, e)
        down = compute(a, f)
        ratio = up/(2.0*down)

        if(ratio > 0.25):
            return 2
        elif(ratio > 0.21 and ratio <= 0.25):
            return 1
        else:
            return 0
            

    _, f = cap.read()
    face_frame = f.copy()
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            face_frame = frame.copy()
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(
                landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(
                landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if(left_blink == 0 or right_blink == 0):
                sleepy += 1
                drowsy = 0
                active = 0
                if(sleepy > 6):
                    playsound('soundfiles/alarm.mp3')
                    status = "SLEEPING!!"
                    color = (255, 0, 0)

            elif (left_blink == 1 or right_blink == 1):
                sleepy = 0
                drowsy += 1
                active = 0
                if(drowsy > 6):
                    playsound('soundfiles/warning.mp3')
                    status = "DROWSY!"
                    color = (0, 0, 255)

            else:
                drowsy = 0
                sleepy = 0
                active += 1
                if(active > 6):
                    status = "ACTIVE:)"
                    color = (0, 255, 0)

            cv2.putText(frame, status, (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

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
