import os
import pytesseract
import cv2
import csv
import time
from distraction_detector import process_screenshot, bounding_box

DATASET = "dataset_name"
DATA_PATH = "" # Path to the folder containing the images
TEXT_PATH = "" # Path to the folder that will contain the text dataset

# Reference the installed location of Tesseract-OCR in your system
# Get Tesseract-OCR from: https://github.com/tesseract-ocr/tesseract
pytesseract.pytesseract.tesseract_cmd = 'tesseract.exe'

# Create the text dataset csv file
csvFile = os.path.join(TEXT_PATH, f"{DATASET}.csv")
with open(csvFile, "w", newline='', encoding='utf-8') as fp:
    wr = csv.writer(fp, delimiter=';')
    wr.writerow(["class", "text", "image"])

def process_image(image_path):
    """
    Process an image and return the text extracted from it.
    @param image_path: Path to the image to be processed
    - Step 1: Process the image to highlight the text
    - Step 2: Detect the bounding boxes from the contours of the text
    - Step 3: Extract the text from the bounding boxes
    - Step 4: Process and return the text
    @return: text
    """

    # Step 1 [preprocessing]
    img = cv2.imread(image_path)
    dilation = process_screenshot(img)

    # Step 2 [detect bounding boxes]
    contours = bounding_box(dilation)

    # Step 3 [extract text]
    im2 = img.copy()
    text_list = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cropped = im2[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)
        text_list.append(text)

    # Step 4 [process text]
    text = " ".join(text_list)
    # remove newlines, tabs
    text = text.replace("\n", " ").replace("\t", " ")
    # remove multiple spaces
    text = " ".join(text.split())
    # remove csv delimiter
    text = text.replace(";", " ")

    return text

if __name__ == '__main__':
    # For each class in the dataset
    for class_name in os.listdir(DATA_PATH)[1:]:
        C = 0
        # For each image in the class
        for image_name in os.listdir(os.path.join(DATA_PATH, class_name)):
            start = time.time()
            C += 1
            print(f"Processing {C}: {image_name}")
            image_path = os.path.join(DATA_PATH, class_name, image_name)
            text = process_image(image_path)
            # Write row to csv file (text dataset)
            csvFile = os.path.join(TEXT_PATH, f"{DATASET}.csv")
            with open(csvFile, "a", newline='', encoding='utf-8') as fp:
                wr = csv.writer(fp, delimiter=';')
                if class_name == "0_focused":
                    wr.writerow([0, text, image_name])
                else:
                    wr.writerow([1, text, image_name])
            end = time.time()
            print(f"Time elapsed (s): {end - start}")
