import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class ImageEmotionDetector:
    '''
    Define a class to encapsulate emotion detection functionality from images.
    '''
    def __init__(self, model_path="emotion_model.h5"):
        '''
        This is the constructor method that initialize the class with a pre-trained model loaded from the specified model path,
        A list of emotion labels corresponding to the model's output.
        :param model_path: emotion_model.h5
        '''
        #Loads the .h5 file containing the trained CNN model.
        self.model = load_model(model_path)
        #Maps numerical indices to human-readable emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def detect_emotions_in_image(self, image_path):
        '''
        Takes the path to an image file and process it to detect emotions.
        :param image_path:
        :return: image
        '''
        image = cv2.imread(image_path) #Reads the image from the specified path into a NumPy array, and returns None if the file doesn't exist or can't be read.
        #Raises a file not found error if the image couldn't be loaded.
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        #Convert the colored image into a grayscale to reduce computations and to be compatible with most pre-trained face detection models.
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Load the face detector by initializing OpenCV's Haar Cascade face detector using pre-trained XML file.
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #Detect faces in greyscale image parameters in which scaleFactor compensate for different face sizes in the image, miniNeighbors are higher values filter false positives, and minSize ensures only faces larger than (30,30) pixels are detected.
        #Faces is a list of bounding box coordinates for each detected face (x,y,w,h).
        faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        #Iterate over each detected face.
        for (x, y, w, h) in faces:
            #Extract the region of interest corresponding to the face.
            face = gray_scale[y:y+h, x:x+w]
            #Rescale the face to 48x48 pixels.
            face_resized = cv2.resize(face, (48, 48))
            #Converts pixel values to the range [0,1].
            face_normalized = face_resized.astype('float32') / 255.0
            #Converts the processed face into a format suitable for TensorFlow/Keras.
            face_array = img_to_array(face_normalized)
            face_array = np.expand_dims(face_array, axis=0)  #Expands the dimension by adding a batch dimension ((1, 48, 48, 1)) for model prediction.
            #Uses the trained model to predict emotion probabilities for the face.
            emotion_probabilities  = self.model.predict(face_array)[0]
            #Finds the index of the highest probability.
            max_index = np.argmax(emotion_probabilities)
            emotion = self.emotion_labels[max_index]  #Retrieve the emotion label corresponding to the highest probability.
            #Draws a green rectangle around the detected faces, then places the emotion label above this rectangle.
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image

    def show_emotions_on_image(self, image):
        '''
        Displays the processed image with emotions.
        :param image:
        :return:
        '''
        #Converts the OpenCV BGR image to RGB format for proper rendering in Matplotlib.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Plot the image containing the detected emotions.
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()




