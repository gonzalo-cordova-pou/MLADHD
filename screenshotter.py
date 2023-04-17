import cv2
import numpy as np
import pytesseract
import time
import os
import pyautogui
from mlmodeling import *

label = "distracted"

# Global variables
INTERVAL = 10 # interval in seconds
SCREENSHOT_COUNTER = 1
DEPLOYMENT_PATH = os.path.join(os.getcwd(), '..', 'screenshots')
CLASSIFIER_PATH = os.path.join(os.getcwd(), '..', 'models')
MODEL = "model_with_extended_dataset_resnet50_2023-03-27_13-23-33"
SESSION_ID = None


model_name = "model_with_extended_dataset"
with open(os.path.join(CLASSIFIER_PATH, MODEL+".json"), "r") as fp:
    hyperparams = json.load(fp)
classifier = classifier = MLADHD(model_name, None, CLASSIFIER_PATH, hyperparams)
classifier.load_model(os.path.join(CLASSIFIER_PATH, MODEL+".pth"))

SCREENSHOT_PATH =  None

def start_session():
    
    global SESSION_ID
    global SCREENSHOT_PATH
    global SCREENSHOT_COUNTER

    if SESSION_ID is None:
        SESSION_ID = time.strftime('%Y%m%d_%H%M%S')
    
    SCREENSHOT_PATH =  os.path.join(DEPLOYMENT_PATH, SESSION_ID, label)
    if not os.path.exists(SCREENSHOT_PATH):
        os.makedirs(SCREENSHOT_PATH)

def take_screenshot():
    global SCREENSHOT_COUNTER
    filename = f"screenshot_{SESSION_ID}_{SCREENSHOT_COUNTER}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    pyautogui.screenshot(os.path.join(SCREENSHOT_PATH, filename))
    SCREENSHOT_COUNTER += 1
    return filename

def distraction_detection(img):
    pred, prob = classifier.predict(img, raw_output=True)
    return pred, prob

def main():
    start_session()
    while True:
        start = time.time()
        print("Screenshot #", SCREENSHOT_COUNTER)
        screenshot_filename = take_screenshot()
        print("Detecting distraction ({}).".format(screenshot_filename))
        pred, prob = distraction_detection(os.path.join(SCREENSHOT_PATH, screenshot_filename))
        if pred == 1:
            print("Predicted: distracted")
        else:
            print("Predicted: focused")
        elapsed = time.time() - start
        sleeping = INTERVAL - elapsed
        print("Time elapsed: ", round(elapsed, 2), "seconds")
        if time.time() - start < INTERVAL:
            print("Sleeping for ", round(sleeping, 2), "seconds")
            time.sleep(INTERVAL - (time.time() - start))

main()