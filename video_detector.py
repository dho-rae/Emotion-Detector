import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

class VideoEmotionDetector:
    '''
    Define a class to encapsulate emotion detection functionality from a real-time video using the webcam.
    '''
    def __init__(self, model_path="emotion_model.h5"):
        '''
        This is the constructor method that initialize the class with a pre-trained model loaded from the specified model path,
        A list of emotion labels corresponding to the model's output.
        :param model_path: emotion_model.h5
        '''
        # Loads the .h5 file containing the trained CNN model.
        self.model = load_model(model_path)
        # Maps numerical indices to human-readable emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def process_face(self, face):
        '''
        Prepares a given face image for emotion prediction by resizing, normalizing, and converting
        it into a format suitable for the model. Returns the processed face image, ready to be passed into the model for prediction.

        :param face:
        :return: face_array
        '''
        #Resize the input face to 48x48 pixels, the size expected by the emotion model.
        face_resized = cv2.resize(face, (48, 48))
        #converts the pixel values to floats and normalizes them to the range [0,1].
        face_normalized = face_resized.astype('float32') / 255.0
        #converts the face image into a Keras-compatible array.
        face_array = img_to_array(face_normalized)
        #Adds an additional dimension to the array, making it a batch of one image.
        face_array = np.expand_dims(face_array,axis=0)

        return face_array

    def detect_emotions_in_video(self):
        '''
        Captures video from the webcam, detects faces in each frame, and displays the emotions
        detected in real-time using OpenCV.
        :return:
        '''
        #Loads the pre-trained Haar Cascade classifier for detecting faces.
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #Start the webcam.
        cap = cv2.VideoCapture(0)
        #Creats a resizable window for displaying the video feed.
        cv2.namedWindow("Real-Time Emotion Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-Time Emotion Detector", 600, 400)
        #Captures video frame in a loop until an exit key is pressed or if the frame capture fails.
        while True:
            ret, frame = cap.read() #ret is a boolean indicating the next frame is successfully captured, cap.read() captures the next frame.
            if not ret:
                print("Failed to capture frame. Exiting...")
                break
            #Flips the frame horizontally for a more natural mirrored view.
            frame = cv2.flip(frame, 1)
            #Converts the frame into grayscale, as the face detector works better on grayscale images.
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Detect faces in the grayscale frame and return the bounding boxes.
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray_frame[y:y+h, x:x+w]  #Extract the face recognition for each detected face.
                #process the face.
                face_array = self.process_face(face)
                #Predict the emotion.
                emotion_probabilities = self.model.predict(face_array)[0]
                #Determine the emotion from the highest probability.
                max_index = np.argmax(emotion_probabilities)
                emotion = self.emotion_labels[max_index]
                # Draws a green rectangle around the detected faces, then places the emotion label above this rectangle.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            #Display the annotated video frame in the window.
            cv2. imshow('Real-Time Emotion Detector', frame)
            #Closes the window when the ESC key is pressed or when the window is closed
            if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Real-Time Emotion Detector", cv2.WND_PROP_VISIBLE) < 1:
                break
        #Releases the webcam and closed the OpenCV window.
        cap.release()
        cv2.destroyAllWindows()

    def detect_emotions_in_video_on_canvas(self, canvas, running_video_callback):
        '''
        Captures video and displays it on a Tkinter canvas, with emotion annotations, using a callback function
        to control whether the video should continue running.
        :param canvas:
        :param running_video_callback:
        :return:
        '''
        #Loads the face detector and star the webcam.
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        def update_frame():
            '''
            Nested function that handles frame-by-frame updates for the canvas.
            :return:
            '''
            #Indicates the video should stop
            if not running_video_callback():
                cap.release()
                return
            #Captures the current frame from the webcam.
            ret, frame = cap.read()
            if not ret:
                return
            #Captures, mirrors and process the frame to detect faces and predict emotions.
            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray_frame[y:y + h, x:x + w]
                face_array = self.process_face(face)

                emotion_probabilities = self.model.predict(face_array)[0]
                max_index = np.argmax(emotion_probabilities)
                emotion = self.emotion_labels[max_index]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            #Converts the OpenCV frame into a format suitable for Tkinter's canvas.
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            canvas_image = ImageTk.PhotoImage(pil_image)
            #Update the canvas
            canvas.delete("all")
            canvas.image = canvas_image
            canvas.create_image(0, 0, anchor=tk.NW, image=canvas_image)
            #Schedules the next frame update in 30 milliseconds.
            canvas.after(30, update_frame)
        #Start frame loop
        update_frame()



