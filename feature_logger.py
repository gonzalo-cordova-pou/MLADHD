import cv2
import numpy as np
import pytesseract
import time
import os
import csv
import concurrent.futures
import pyautogui
from mlmodeling import *
import winsound

frequency1 = 2500  # Set Frequency To 2500 Hertz, 
frequency2 = 200  # Set Frequency To 1000 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second


# Global variables
INTERVAL = None # interval in seconds, None for no interval
SCREENSHOT_COUNTER = 1
DEPLOYMENT_PATH = os.path.join(os.getcwd(), '..', 'deployment')
CLASSIFIER_PATH = os.path.join(os.getcwd(), '..', 'models')
MODEL = "model_with_extended_dataset_resnet50_2023-04-20_11-24-17"
SESSION_ID = None
LOG_PATH = None

model_name = "model_with_extended_dataset"

hyperparams = None
classifier = None

SCREENSHOT_PATH =  None

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = "E:\\Users\\ADHD Project\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

def start_session():
    
    global SESSION_ID
    global SCREENSHOT_PATH
    global LOG_PATH
    global SCREENSHOT_COUNTER
    global hyperparams
    global classifier

    with open(os.path.join(CLASSIFIER_PATH, MODEL+".json"), "r") as fp:
        hyperparams = json.load(fp)
    classifier = MLADHD(model_name, None, CLASSIFIER_PATH, hyperparams)
    classifier.load_model(os.path.join(CLASSIFIER_PATH, MODEL+".pth"))

    if SESSION_ID is None:
        SESSION_ID = time.strftime('%Y%m%d_%H%M%S')
    
    SCREENSHOT_PATH =  os.path.join(DEPLOYMENT_PATH, SESSION_ID, "screenshots")
    if not os.path.exists(SCREENSHOT_PATH):
        os.makedirs(SCREENSHOT_PATH)
    
    LOG_PATH = os.path.join(DEPLOYMENT_PATH, SESSION_ID, "log")
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    with open(os.path.join(LOG_PATH, f"{SESSION_ID}.csv"), "w", newline='', encoding='utf-8') as fp:
        wr = csv.writer(fp, delimiter=';')
        wr.writerow(["timestamp", "screenshot", "text", "prediction", "probability"])

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

def log(log_dir, screenshot_filename, text, pred, prob):
    # write a line to the log file
    with open(os.path.join(log_dir, f"{SESSION_ID}.csv"), "a", newline='', encoding='utf-8') as fp:
        wr = csv.writer(fp, delimiter=';')
        wr.writerow([time.strftime('%Y%m%d_%H%M%S'), screenshot_filename, text, pred, prob])

def main():
    start_session()
    while True:
        start = time.time()
        print("Screenshot #", SCREENSHOT_COUNTER)
        screenshot_filename = take_screenshot()
        print("Processing screenshot ({}).".format(screenshot_filename))
        img = cv2.imread(os.path.join(SCREENSHOT_PATH, screenshot_filename))
        dilated_img = process_screenshot(img)
        contours = bounding_box(dilated_img)
        text = OCR(img, contours)
        print("Text extracted")
        pred, prob = distraction_detection(os.path.join(SCREENSHOT_PATH, screenshot_filename))
        if pred == 1:
            winsound.Beep(frequency1, duration)
            print("Predicted: distracted")
        else:
            winsound.Beep(frequency2, duration)
            print("Predicted: focused")
        
        log(LOG_PATH, screenshot_filename, text, pred, prob)

        elapsed = time.time() - start
        print("Logged | Time elapsed: ", round(elapsed, 2), "seconds")
        if INTERVAL is not None and time.time() - start < INTERVAL:
            sleeping = INTERVAL - elapsed
            print("Sleeping for ", round(sleeping, 2), "seconds")
            time.sleep(INTERVAL - (time.time() - start))

main()