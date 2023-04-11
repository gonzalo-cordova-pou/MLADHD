import time
import os
import pyautogui

"""
This script takes screenshots of the screen every 'interval' seconds.
"""

interval = 10 # interval in seconds
save_path = 'screenshots/' # path to save screenshots
screenshot_counter = 1

if not os.path.exists(save_path):
    os.makedirs(save_path)

while True:
    # capture the screen and save the screenshot as a JPG file
    filename = f"screenshot_{screenshot_counter}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    pyautogui.screenshot(os.path.join(save_path, filename), format='jpg')

    # increment the screenshot counter
    screenshot_counter += 1

    # wait for 'interval' seconds before taking the next screenshot
    time.sleep(interval)
