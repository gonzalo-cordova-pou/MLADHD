import cv2
import numpy as np
import pytesseract
import time
import os
import pyautogui
from CNNclassifier import *
import winsound

LABEL = "distracted"
FREQ1 = 500      # Set Frequency To 2500 Hertz (distracted)
FREQ2 = 150      # Set Frequency To 1500 Hertz (focused)
DURATION = 1000  # Set sound duration (1000 ms == 1 second)
INTERVAL = 5     # interval between screenshots (seconds)
DEPLOYMENT_PATH = os.path.join(os.getcwd(), '..', 'screenshots')
CLASSIFIER_PATH = os.path.join(os.getcwd(), '..', 'models')
MODEL = "model_with_extended_dataset_resnet50_2023-03-27_13-23-33"
MODEL_NAME = "model_with_extended_dataset"

screenshot_path = None
screenshot_counter = 1
session_id = None

# Load classifier
model_hyperparam_path = os.path.join(CLASSIFIER_PATH, MODEL+".json")
with open(model_hyperparam_path, "r", encoding="utf-8") as fp:
    hyperparams = json.load(fp)
classifier = classifier = MLADHD(MODEL_NAME, None, CLASSIFIER_PATH, hyperparams)
classifier.load_model(os.path.join(CLASSIFIER_PATH, MODEL+".pth"))

def start_session():
    """
    Creates a new session folder for storing classified screenshots as part of
    the Human-in-the-loop (HITL) process.
    """

    global screenshot_path
    global session_id
    global screenshot_counter

    if session_id is None:
        session_id = LABEL + '_' + time.strftime('%Y%m%d_%H%M%S')

    screenshot_path =  os.path.join(DEPLOYMENT_PATH, session_id)
    if not os.path.exists(screenshot_path):
        os.makedirs(screenshot_path)

    if not os.path.exists(os.path.join(screenshot_path, "classified_Distracted")):
        os.makedirs(os.path.join(screenshot_path, "classified_Distracted"))

    if not os.path.exists(os.path.join(screenshot_path, "classified_Focused")):
        os.makedirs(os.path.join(screenshot_path, "classified_Focused"))

def take_screenshot():

    global screenshot_counter

    filename = f"screenshot_{session_id}_{screenshot_counter}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    pyautogui.screenshot(os.path.join(screenshot_path, filename))
    screenshot_counter += 1
    return filename

def main():
    start_session()
    while True:
        start = time.time()
        print("Screenshot #", screenshot_counter)
        screenshot_filename = take_screenshot()
        print(f"Detecting distraction ({screenshot_filename}).")
        pred, _ = classifier.predict(
            os.path.join(screenshot_path, screenshot_filename),
            raw_output=True
        )

        # Intervention (sound) if distracted
        if pred == 1:
            winsound.Beep(FREQ1, DURATION)
            print("Predicted: distracted")
            os.rename(
                os.path.join(screenshot_path, screenshot_filename),
                os.path.join(screenshot_path,"classified_Distracted", screenshot_filename)
            )
        else:
            winsound.Beep(FREQ2, DURATION)
            print("Predicted: focused")
            os.rename(
                os.path.join(screenshot_path, screenshot_filename),
                os.path.join(screenshot_path, "classified_Focused", screenshot_filename)
            )

        # Sleep for the remaining time to ensure INTERVAL is met
        elapsed = time.time() - start
        sleeping = INTERVAL - elapsed
        print("Time elapsed: ", round(elapsed, 2), "seconds")
        if time.time() - start < INTERVAL:
            print("Sleeping for ", round(sleeping, 2), "seconds")
            time.sleep(INTERVAL - (time.time() - start))

main()
