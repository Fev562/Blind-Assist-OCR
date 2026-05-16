import pytesseract as tess
from PIL import Image
import cv2
import pyttsx3
import keyboard


engine = pyttsx3.init()

# Set Tesseract path 
tess.pytesseract.tesseract_cmd = r'C:\Users\adis\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def capture_image():
    """Captures an image from the camera when 'r' is pressed."""
    cap = cv2.VideoCapture(0)  # 0 for default camera

    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera Preview", frame)

        if keyboard.is_pressed('r'):
            cv2.imwrite("img.png", frame)
            print("Image captured!")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(img_path):
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding for better text extraction
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # Optimized noise reduction
    thresh = cv2.fastNlMeansDenoising(thresh, h=3, templateWindowSize=7, searchWindowSize=21)

    return Image.fromarray(thresh)

def read_text_from_image():
    
    img_path = "C:\\Users\\adis\\desktop\\OptiText\\6.png"
    preprocessed_img = preprocess_image(img_path)
    text = tess.image_to_string(preprocessed_img, config='--psm 6')  # Page segmentation mode 6 for single block

    # Set volume slightly higher
    volume = engine.getProperty('volume')
    engine.setProperty('volume', min(volume + 0.1, 1)) 
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    print(text)
    engine.runAndWait()
    engine.stop()

if __name__ == "__main__":
    #capture_image()  
    read_text_from_image()  


