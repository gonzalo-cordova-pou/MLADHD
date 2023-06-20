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

frequency1 = 2500  # Set Frequency To 2500 Hertz, 
frequency2 = 200  # Set Frequency To 1000 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second


# Global variables
INTERVAL = 3 # interval in seconds, None for no interval
WINDOW = 3 # look at the last {WINDOW} screenshots for intervention
LAST_PREDICTIONS = []

SCREENSHOT_COUNTER = 1
DEPLOYMENT_PATH = os.path.join(os.getcwd(), '..', 'deployment')
CLASSIFIER_PATH = os.path.join(os.getcwd(), '..', 'models')
CNN_MODEL = "resnet50_xlarge_resnet50_2023-05-19_21-41-02"
NLP_MODEL = "runs:/8747e2d5b450477eab8336fbd91179e2/model" # Run id mlflow
SESSION_ID = None
LOG_PATH = None

model_name = "model_with_extended_dataset"

hyperparams = None
classifier = None

SCREENSHOT_PATH =  None
FOCUSED_PATH = None
DISTRACTED_PATH = None

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = "tesseract.exe"

def start_session():
    
    global SESSION_ID
    global SCREENSHOT_PATH
    global FOCUSED_PATH
    global DISTRACTED_PATH
    global LOG_PATH
    global SCREENSHOT_COUNTER
    global hyperparams
    global classifier

    if SESSION_ID is None:
        SESSION_ID = time.strftime('%Y%m%d_%H%M%S')
    
    SCREENSHOT_PATH =  os.path.join(DEPLOYMENT_PATH, SESSION_ID, "screenshots")
    if not os.path.exists(SCREENSHOT_PATH):
        os.makedirs(SCREENSHOT_PATH)
    
    # make a dir for focused and distracted screenshots
    FOCUSED_PATH = os.path.join(SCREENSHOT_PATH, "focused")
    if not os.path.exists(FOCUSED_PATH):
        os.makedirs(FOCUSED_PATH)
    DISTRACTED_PATH = os.path.join(SCREENSHOT_PATH, "distracted")
    if not os.path.exists(DISTRACTED_PATH):
        os.makedirs(DISTRACTED_PATH)
    
    LOG_PATH = os.path.join(DEPLOYMENT_PATH, SESSION_ID, "log")
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    with open(os.path.join(LOG_PATH, f"{SESSION_ID}.csv"), "w", newline='', encoding='utf-8') as fp:
        wr = csv.writer(fp, delimiter=';')
        wr.writerow(["timestamp", "screenshot", "text", "prediction", "probability", "type"])

def take_screenshot():
    global SCREENSHOT_COUNTER
    filename = f"screenshot_{SESSION_ID}_{SCREENSHOT_COUNTER}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    pyautogui.screenshot(os.path.join(SCREENSHOT_PATH, filename))
    SCREENSHOT_COUNTER += 1
    return filename

def process_screenshot(img):

    # Crop 20px from each side of the image, except 60px from the bottom
    img = img[0:img.shape[0] - 60, 20:img.shape[1] - 20]

    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 2)

    return dilation

def bounding_box(img):
    # Finding contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
    # Sort all the contours avoiding annotations
    # by x-coordinate (left to right), then y-coordinate (top to bottom), then area (big to small)
    #lambda_sort = lambda x: (cv2.boundingRect(x)[0], cv2.boundingRect(x)[1], cv2.contourArea(x))
    #contours = sorted(contours, key=lambda_sort)
    return contours

def OCR(img, contours):
    
    im2 = img.copy()

    text_list = []

    def contour2text(contour):
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

    text = " ".join(text_list)
    # remove newlines, tabs
    text = text.replace("\n", " ").replace("\t", " ")
    # remove multiple spaces
    text = " ".join(text.split())
    # remove csv delimiter
    text = text.replace(";", " ")
    return text

def distraction_detection(img):
    pred, prob = classifier.predict(img, raw_output=True)
    return pred, prob

def log(log_dir, screenshot_filename, text, pred, prob, type):
    # write a line to the log file
    with open(os.path.join(log_dir, f"{SESSION_ID}.csv"), "a", newline='', encoding='utf-8') as fp:
        wr = csv.writer(fp, delimiter=';')
        wr.writerow([time.strftime('%Y%m%d_%H%M%S'), screenshot_filename, text, pred, prob, type])

def main(argv):
    global classifier
    global LAST_PREDICTIONS
    # cnn: use the CNN model for distraction detection, no OCR is performed
    # text: use the text model for distraction detection, OCR is performed on the screenshots
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
            classifier = MLADHD(model_name, None, CLASSIFIER_PATH, hyperparams)
            classifier.load_model(os.path.join(CLASSIFIER_PATH, CNN_MODEL+".pth"))
    print("Starting session with {} model.".format(mode))
    start_session()
    
    # Start the session loop. Take screenshot -> process -> predict -> log -> repeat
    while True:
        start = time.time()
        print("Screenshot #", SCREENSHOT_COUNTER)
        screenshot_filename = take_screenshot()
        print("Processing screenshot ({}).".format(screenshot_filename))
        
        # Classify the screenshot
        if mode == "cnn":
            pred, prob = distraction_detection(os.path.join(SCREENSHOT_PATH, screenshot_filename))
            LAST_PREDICTIONS.append(pred)
            if len(LAST_PREDICTIONS) > WINDOW:
                LAST_PREDICTIONS.pop(0)
            text = None
        else:
            
            # Read -> process -> OCR -> predict
            img = cv2.imread(os.path.join(SCREENSHOT_PATH, screenshot_filename))
            dilated_img = process_screenshot(img)
            contours = bounding_box(dilated_img)
            text = OCR(img, contours)
            pred = sk_model.predict([text])
            pred = pred[0]
            prob = None
            print("Text extracted")
        
        # Move the screenshot to the focused or distracted folder based on the prediction
        if pred == 0:
            shutil.move(os.path.join(SCREENSHOT_PATH, screenshot_filename), os.path.join(FOCUSED_PATH, screenshot_filename))
        else:
            shutil.move(os.path.join(SCREENSHOT_PATH, screenshot_filename), os.path.join(DISTRACTED_PATH, screenshot_filename))
        
        # INTERVENTION: if the last {WINDOW} predictions are distracted, play a sound
        if len(LAST_PREDICTIONS) == WINDOW and LAST_PREDICTIONS[ : WINDOW] == [1] * WINDOW:
            winsound.Beep(frequency2, duration)
            print("Predicted: distracted")
            # to avoid multiple beeps
            LAST_PREDICTIONS = [0] * WINDOW
        else:
            # winsound.Beep(frequency1, duration)
            print("Predicted: focused")
        

        
        # Log the prediction
        log(LOG_PATH, screenshot_filename, text, pred, prob, mode)
        
        # Sleep for the remaining time of the interval (if any)
        elapsed = time.time() - start
        print("Logged | Time elapsed: ", round(elapsed, 2), "seconds")
        if INTERVAL is not None and time.time() - start < INTERVAL:
            sleeping = INTERVAL - elapsed
            print("Sleeping for ", round(sleeping, 2), "seconds")
            time.sleep(INTERVAL - (time.time() - start))

if __name__ == "__main__":
   main(sys.argv[1:])