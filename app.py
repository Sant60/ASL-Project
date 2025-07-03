from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)

# Initialize camera and models
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", " SPECIAL THANKS TO SAIMA MAAM"
]

# Image preprocessing constants
offset = 20
imgSize = 300

# State variables
current_letter = ""
confidence = 0.0
word = ""
last_letter = ""
letter_start_time = None
auto_add_delay = 5  # seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_letter', methods=['POST'])
def add_letter():
    global word, current_letter
    if current_letter:
        word += current_letter
    return "OK"

@app.route('/undo_letter', methods=['POST'])
def undo_letter():
    global word
    if len(word) > 0:
        word = word[:-1]
    return "OK"

@app.route('/reset', methods=['POST'])
def reset():
    global word
    word = ""
    return "OK"

@app.route('/get_word')
def get_word():
    return jsonify({"word": word})

def generate_frames():
    global word, current_letter, confidence, last_letter, letter_start_time

    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            try:
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                current_letter = labels[index]
                confidence = prediction[index]

                if current_letter == last_letter:
                    if letter_start_time and time.time() - letter_start_time >= auto_add_delay:
                        word += current_letter
                        letter_start_time = None
                        last_letter = ""
                else:
                    last_letter = current_letter
                    letter_start_time = time.time()

                # Show predicted letter and confidence
                text_display = f"{current_letter} ({int(confidence * 100)}%)"
                cv2.rectangle(imgOutput, (x - offset, y - offset - 60),
                              (x - offset + 160, y - offset - 10), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, text_display, (x - offset + 5, y - offset - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

            except:
                current_letter = ""
                confidence = 0.0
                last_letter = ""
                letter_start_time = None
        else:
            current_letter = ""
            confidence = 0.0
            last_letter = ""
            letter_start_time = None

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(debug=True)