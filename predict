import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load your trained model (adjust filename/path)
model = load_model('my_model.keras')

# Define class labels and their emojis
class_labels = ['palm', 'fist', 'thumb', 'ok', 'index', 'l', 'fist_moved', 'palm_moved', 'c', 'down']
emoji_map = {
    'palm': '🖐️',
    'fist': '✊',
    'thumb': '👍',
    'ok': '👌',
    'index': '☝️',
    'l': '🤟',
    'fist_moved': '✊🏽',
    'palm_moved': '🖐🏽',
    'c': '🤙',
    'down': '👇'
}

IMAGE_SIZE = 64  # Adjust according to your model input

def preprocess_image(img):
    # Convert to grayscale if your model expects 1 channel
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # add channel dimension
    img = np.expand_dims(img, axis=0)   # add batch dimension
    return img

def predict_image(img):
    processed = preprocess_image(img)
    predictions = model.predict(processed)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

def display_result(img, predicted_class, confidence):
    emoji = emoji_map.get(predicted_class, '')
    text = f"{predicted_class} {emoji} ({confidence*100:.1f}%)"
    # Put text on the image
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Prediction", img)

def live_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        predicted_class, confidence = predict_image(frame)
        display_result(frame, predicted_class, confidence)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def upload_and_predict():
    Tk().withdraw()  # Close the root window
    file_path = askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        print("No file selected")
        return
    img = cv2.imread(file_path)
    if img is None:
        print("Failed to load image")
        return
    predicted_class, confidence = predict_image(img)
    display_result(img, predicted_class, confidence)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    choice = input("Choose option:\n1 - Upload Image\n2 - Live Camera\nEnter choice: ")
    if choice == '1':
        upload_and_predict()
    elif choice == '2':
        live_camera()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
