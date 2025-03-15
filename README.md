# **Emotion Detector - Facial Expression Recognition**

## **Overview**  
Emotion Detector is a Python-based application that identifies facial expressions in **images and real-time video**.  
Using a deep learning model trained on the **FER-2013 dataset**, it classifies faces into seven distinct emotions:  
**Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.  

## **How It Works**  
1. The program processes facial images using **OpenCV** for detection.  
2. A **CNN model (TensorFlow/Keras)** predicts the most likely emotion.  
3. The program provides feedback via a **graphical user interface (Tkinter)** or overlays results on real-time video streams.  

## **Key Features**  
- **Real-Time Emotion Recognition** – Detects expressions in live webcam feeds.  
- **Static Image Analysis** – Classifies emotions from uploaded images.  
- **Deep Learning Model** – Trained on the **FER-2013 dataset** for high accuracy.  
- **GUI-Based Interaction** – Simple and interactive Tkinter interface.  
- **Data Processing & Visualization** – Uses **Pandas, NumPy, and Pillow**.  

## **Running the Program**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/emotion-detector.git
cd emotion-detector
```

### **2. Install Dependencies**  
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### **3. Run Emotion Detection**  

#### **For real-time video detection:**
```bash
python detect_emotion_video.py
```

#### **For image-based detection:**
```bash
python detect_emotion_image.py --image path_to_image.jpg
```

## **Example Output**
When running the real-time video detection, the program will display the live webcam feed with an overlay showing the detected emotion.
For an image file, the console output might look like this:
```bash
Processing image: test_image.jpg  
Predicted Emotion: Happy 😊  
```

## **Tech Stack**
Programming Language: Python
Deep Learning Framework: TensorFlow/Keras
Computer Vision: OpenCV
GUI Framework: Tkinter
Data Handling: Pandas, NumPy, Pillow
