import cv2
import numpy as np
import pytesseract
import time
import os
import pyautogui
from CNNclassifier import *
import winsound

label = "distracted"

frequency1 = 500  # Set Frequency To 2500 Hertz, 
frequency2 = 150  # Set Frequency To 1000 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

# Global variables
INTERVAL = 5 # interval in seconds
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
        SESSION_ID = label + '_' + time.strftime('%Y%m%d_%H%M%S')
    
    SCREENSHOT_PATH =  os.path.join(DEPLOYMENT_PATH, SESSION_ID)
    if not os.path.exists(SCREENSHOT_PATH):
        os.makedirs(SCREENSHOT_PATH)
    
    if not os.path.exists(os.path.join(SCREENSHOT_PATH, "classified_Distracted")):
        os.makedirs(os.path.join(SCREENSHOT_PATH, "classified_Distracted"))
    
    if not os.path.exists(os.path.join(SCREENSHOT_PATH, "classified_Focused")):
        os.makedirs(os.path.join(SCREENSHOT_PATH, "classified_Focused"))

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
            winsound.Beep(frequency1, duration)
            print("Predicted: distracted")
            os.rename(os.path.join(SCREENSHOT_PATH, screenshot_filename), os.path.join(SCREENSHOT_PATH, "classified_Distracted", screenshot_filename))
        else:
            winsound.Beep(frequency2, duration)
            print("Predicted: focused")
            os.rename(os.path.join(SCREENSHOT_PATH, screenshot_filename), os.path.join(SCREENSHOT_PATH, "classified_Focused", screenshot_filename))
        elapsed = time.time() - start
        sleeping = INTERVAL - elapsed
        print("Time elapsed: ", round(elapsed, 2), "seconds")
        if time.time() - start < INTERVAL:
            print("Sleeping for ", round(sleeping, 2), "seconds")
            time.sleep(INTERVAL - (time.time() - start))

main()