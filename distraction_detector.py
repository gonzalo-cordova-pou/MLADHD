import cv2
import numpy as np
import pytesseract
import time
import os
import csv
import sys
import shutil
import concurrent.futures
import pyautogui
from CNNclassifier import *
import winsound


FREQ = 200       # Set Frequency To 1000 Hertz
DURATION = 1000  # Set sound duration (1000 ms == 1 second)
INTERVAL = 5     # interval between screenshots (seconds)
WINDOW = 3       # look at the last {WINDOW} screenshots for intervention
DEPLOYMENT_PATH = os.path.join(os.getcwd(), '..', 'deployment')
CLASSIFIER_PATH = os.path.join(os.getcwd(), '..', 'models')
CNN_MODEL = "resnet50_xlarge_resnet50_2023-05-19_21-41-02"
NLP_MODEL = "runs:/8747e2d5b450477eab8336fbd91179e2/model" # Run id mlflow
MODEL_NAME = "model_with_extended_dataset"

last_predictions = []
screenshot_counter = 1
session_id = None
hyperparams = None
classifier = None
screenshot_path = None
log_path = None
focused_path = None
distracted_path = None

# Reference the installed location of Tesseract-OCR in your system
# Get Tesseract-OCR from: https://github.com/tesseract-ocr/tesseract
pytesseract.pytesseract.tesseract_cmd = "tesseract.exe"

def start_session():
    """
    Creates a new session folder for storing classified screenshots as part of
    the Human-in-the-loop (HITL) process.
    """

    global session_id
    global screenshot_path
    global focused_path
    global distracted_path
    global log_path
    global screenshot_counter
    global hyperparams
    global classifier

    if session_id is None:
        session_id = time.strftime('%Y%m%d_%H%M%S')

    # make a dir for focused and distracted screenshots
    screenshot_path =  os.path.join(DEPLOYMENT_PATH, session_id, "screenshots")
    if not os.path.exists(screenshot_path):
        os.makedirs(screenshot_path)
    focused_path = os.path.join(screenshot_path, "focused")
    if not os.path.exists(focused_path):
        os.makedirs(focused_path)
    distracted_path = os.path.join(screenshot_path, "distracted")
    if not os.path.exists(distracted_path):
        os.makedirs(distracted_path)

    # make a dir for the log file
    log_path = os.path.join(DEPLOYMENT_PATH, session_id, "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # create the log file and write the header (csv)
    session_log_path = os.path.join(log_path, f"{session_id}.csv")
    with open(session_log_path, "w", newline='', encoding='utf-8') as fp:
        wr = csv.writer(fp, delimiter=';')
        wr.writerow(["timestamp", "screenshot", "text", "prediction", "probability", "type"])

def take_screenshot():

    global screenshot_counter

    filename = f"screenshot_{session_id}_{screenshot_counter}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    pyautogui.screenshot(os.path.join(screenshot_path, filename))
    screenshot_counter += 1
    return filename

def process_screenshot(img):
    """
    Preprocess the screenshot for OCR: crop, grayscale, blur, threshold, dilate
    """

    # Crop 20px from each side of the image, except 60px from the bottom
    img = img[0:img.shape[0] - 60, 20:img.shape[1] - 20]

    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian Blur
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # Performing OTSU threshold
    _, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # Define the shape and size of the structure and kernel.
    # The kernel size determines the area of the rectangle used for detection.
    # Using a smaller value such as (10, 10) will detect individual words
    # rather than whole sentences.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 2)

    return dilation

def bounding_box(img):
    """
    Find text bounding boxes
    prereq: img has been processed with process_screenshot()
    """

    contours, _ = cv2.findContours(
                        img,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE
                    )
    return contours

def OCR(img, contours):
    """
    Extract text from the screenshot
    @param img: the screenshot
    @param contours: the contours of the screenshot (bounding boxes)
    """

    im2 = img.copy()
    text_list = []

    def contour2text(contour):
        """
        Use pytesseract to extract text from a bounding box
        """
        x, y, w, h = cv2.boundingRect(contour)
        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]
        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)
        return text

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    
    # parallelize the OCR
    with concurrent.futures.ThreadPoolExecutor() as executor:
        text_list = executor.map(contour2text, contours)

    # Text preprocessing
    text = " ".join(text_list)
    # remove newlines, tabs
    text = text.replace("\n", " ").replace("\t", " ")
    # remove multiple spaces
    text = " ".join(text.split())
    # remove csv delimiter
    text = text.replace(";", " ")
    return text

def log(log_dir, screenshot_filename, text, pred, prob, type):
    """
    Write a row to the log file with one screenshot's data
    """

    session_log_dir = os.path.join(log_dir, f"{session_id}.csv")
    with open(session_log_dir, "a", newline='', encoding='utf-8') as fp:
        wr = csv.writer(fp, delimiter=';')
        wr.writerow(
            [
                time.strftime('%Y%m%d_%H%M%S'),
                screenshot_filename,
                text, # extracted text
                pred, # prediction
                prob, # probability of the prediction
                type  # cnn or text
            ]
        )

def main(argv):
    """
    argv: cnn (default) or text
    if cnn: use the CNN model for distraction detection, no OCR is performed
    if text: use the text model for distraction detection, OCR is performed on
             the screenshots
    """

    global classifier
    global last_predictions
    
    # Parse arguments and load model. If no argument is given, use CNN by default
    if len(argv) == 1:
        if argv[0] == "cnn":
            mode = "cnn"
        elif argv[0] == "text":
            mode = "text"
            sk_model = mlflow.sklearn.load_model(NLP_MODEL)
        else:
            print("Invalid argument. Usage: python screenshotter.py [cnn|text]")
            sys.exit(2)
    elif len(argv) == 0:
        mode = "cnn"
    else:
        print("Invalid argument. Usage: python screenshotter.py [cnn|text]")
        sys.exit(2)
    if mode == "cnn":
        with open(os.path.join(CLASSIFIER_PATH, CNN_MODEL+".json"), "r") as fp:
            hyperparams = json.load(fp)
            classifier = MLADHD(MODEL_NAME, None, CLASSIFIER_PATH, hyperparams)
            classifier.load_model(os.path.join(CLASSIFIER_PATH, CNN_MODEL+".pth"))
    print(f"Starting session with {mode} model.")
    start_session()

    # Start the session loop. Take screenshot -> process -> predict -> log -> repeat
    while True:
        start = time.time()
        print("Screenshot #", screenshot_counter)
        screenshot_filename = take_screenshot()
        print(f"Processing screenshot ({screenshot_filename}).")
        # Extract text from the screenshot (if mode == "text")
        # Predict distraction (both modes)
        if mode == "cnn":
            pred, prob = classifier.predict(
                os.path.join(screenshot_path, screenshot_filename),
                raw_output=True
            )
            last_predictions.append(pred)
            if len(last_predictions) > WINDOW:
                last_predictions.pop(0)
            text = None
        else:
            # Read -> process -> OCR -> predict
            img = cv2.imread(os.path.join(screenshot_path, screenshot_filename))
            dilated_img = process_screenshot(img)
            contours = bounding_box(dilated_img)
            text = OCR(img, contours)
            pred = sk_model.predict([text])
            pred = pred[0]
            prob = None
            print("Text extracted")
        
        # Move the screenshot to the focused or distracted folder based on the prediction
        # As part of the Human-in-the-loop (HITL) process
        if pred == 0:
            shutil.move(os.path.join(screenshot_path, screenshot_filename), os.path.join(focused_path, screenshot_filename))
        else:
            shutil.move(os.path.join(screenshot_path, screenshot_filename), os.path.join(distracted_path, screenshot_filename))
        
        # INTERVENTION: if the last {WINDOW} predictions are distracted, play a sound
        if len(last_predictions) == WINDOW and last_predictions[ : WINDOW] == [1] * WINDOW:
            winsound.Beep(FREQ, DURATION)
            print("Predicted: distracted")
            # to avoid multiple beeps
            last_predictions = [0] * WINDOW
        else:
            print("Predicted: focused")

        # Log the prediction
        log(log_path, screenshot_filename, text, pred, prob, mode)

        # Sleep for the remaining time to ensure INTERVAL is met
        elapsed = time.time() - start
        print("Logged | Time elapsed: ", round(elapsed, 2), "seconds")
        if INTERVAL is not None and time.time() - start < INTERVAL:
            sleeping = INTERVAL - elapsed
            print("Sleeping for ", round(sleeping, 2), "seconds")
            time.sleep(INTERVAL - (time.time() - start))

if __name__ == "__main__":
   main(sys.argv[1:])
