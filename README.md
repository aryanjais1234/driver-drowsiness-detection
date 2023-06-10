# Drowsiness Detection using Facial Landmarks

This is a Python-based web application that detects drowsiness using facial landmarks. It utilizes computer vision techniques and a trained deep learning model to detect eyes and classify them as either open or closed. If closed eyes are detected for a certain duration, an alarm sound is played to alert the user.

## Functionality

The application performs the following steps:

1. Captures video frames from the webcam using OpenCV.
2. Detects faces in the video frames using the Haar cascade classifier.
3. Detects eyes within each face region using the Haar cascade classifier.
4. Extracts and preprocesses eye regions for classification.
5. Feeds the preprocessed eye regions into a trained deep learning model.
6. Classifies the eyes as open or closed based on the model's predictions.
7. Keeps track of the number of closed eye frames to determine drowsiness.
8. Displays the video frames with bounding boxes and text indicating the eye status and drowsiness score.
9. Plays an alarm sound when drowsiness is detected for a certain duration.

## Technologies Used

The application is built using the following technologies:

- Python: The programming language used for the application.
- OpenCV: Used for capturing video frames and performing computer vision tasks.
- TensorFlow: Used for loading the pre-trained deep learning model and making predictions.
- Flask: Used to create a web server and serve the application as a web page.
- HTML and CSS: Used for creating the web interface and styling.
- JavaScript: Used for dynamic behavior on the web page.

## Installation and Setup

Follow these steps to run the application:

1. Clone or download the project files to your local machine.
2. Install the required Python packages by running the following command:
   ```
   pip install -r requirements.txt
   ```
3. Download the Haar cascade classifier files from the OpenCV repository:
   - [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
   - [haarcascade_eye.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)
   Place these files in the same directory as the Python files.
4. Download the pre-trained model file ([model.h5](models/model.h5)) and place it in the "models" directory.
5. Download the alarm sound file ([alarm.mp3](alarm.mp3)) and place it in the project directory.
6. Run the application by executing the following command:
   ```
   python app.py
   ```
7. Open your web browser and visit `http://localhost:5000` to access the application.

## Usage

Once the application is running, follow these steps to use it:

1. The web page will display a video stream from your webcam along with bounding boxes and text indicating eye status and drowsiness score.
2. If the eyes are classified as closed for a certain duration (indicating drowsiness), an alarm sound will play to alert the user.
3. To stop the application, press `Ctrl+C` in the terminal.

## Future Improvements

- Implement a more advanced face detection algorithm to handle various face orientations and lighting conditions.
- Fine-tune the model with more diverse eye images for improved accuracy.
- Add additional features such as driver fatigue detection, head pose estimation, and gaze tracking.
- Optimize the code for better performance on different hardware configurations.
- Enhance the user interface with more intuitive controls and feedback.


## References

- [OpenCV Documentation](https://docs.opencv.org

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Haar Cascade Classifiers](https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)
- [Drowsiness Detection using OpenCV and Deep Learning](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
