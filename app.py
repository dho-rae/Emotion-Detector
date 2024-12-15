import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
from image_detector import ImageEmotionDetector
from video_detector import VideoEmotionDetector

class EmotionDetectorGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Detector")
        self.root.geometry("800x600")

        self.image_detector = ImageEmotionDetector()
        self.video_detector = VideoEmotionDetector()

        self.mode = tk.StringVar(value="Image")
        self.running_video = False
        self.video_thread = None

        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="white")
        self.canvas.pack(pady=20)

        tk.Radiobutton(self.root, text="Image", variable=self.mode, value="Image", command=self.select_mode).pack(side="left", padx=80)
        tk.Radiobutton(self.root, text="Video", variable=self.mode, value="Video", command=self.select_mode).pack(side="right", padx=80)

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.start_video_button = tk.Button(self.root, text="Start Video", command=self.start_video)

        self.select_mode()

    def select_mode(self):
        self.canvas.delete("all")

        if self.running_video:
            self.stop_video()

        if self.mode.get() == "Image":
            self.upload_button.pack(side="bottom", pady=10)
            self.start_video_button.pack_forget()
        elif self.mode.get() == "Video":
            self.start_video_button.pack(side="bottom", pady=10)
            self.upload_button.pack_forget()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            processed_image = self.image_detector.detect_emotions_in_image(file_path)
            self.display_image(processed_image)

    def display_image(self, image):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        image_width, image_height = pil_image.size
        aspect_ratio = image_width / image_height

        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)

        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        canvas_image = ImageTk.PhotoImage(pil_image)

        x_coord = (canvas_width - new_width) // 2
        y_coord = (canvas_height - new_height) // 2

        self.canvas.delete("all")
        self.canvas.image = canvas_image
        self.canvas.create_image(x_coord, y_coord, anchor=tk.NW, image=canvas_image)

    def start_video(self):
        self.running_video = True
        threading.Thread(target=self.run_video).start()

    def stop_video(self):
        if self.running_video:
            self.running_video = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join()  # Wait for the video thread to stop
            self.video_thread = None

    def run_video(self):
        self.video_detector.detect_emotions_in_video_on_canvas(self.canvas, lambda: self.running_video)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    application = EmotionDetectorGui()
    application.run()

