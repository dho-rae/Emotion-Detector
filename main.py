from image_detector import  ImageEmotionDetector
from video_detector import VideoEmotionDetector
from app import EmotionDetectorGui

def image_detector_test():
    image_detector = ImageEmotionDetector()
    image_path = "emotion2.jpg"
    processed_image = image_detector.detect_emotions_in_image(image_path)
    image_detector.show_emotions_on_image(processed_image)

def video_detector_test():
    videoDetector = VideoEmotionDetector()
    videoDetector.detect_emotions_in_video()

def run_application():
    application = EmotionDetectorGui()
    application.run()

if __name__ == "__main__":
    run_application()