import os
import pytesseract
import cv2
import csv
import time
from feature_logger import process_screenshot, bounding_box, OCR

dataset = "data_clean_large"
limit_per_class = 4292

# Path to the folder containing the images
dataPath = os.path.join('E:\\mladhd\\datasets\\', dataset)

# Path to the folder that will contain the text dataset
textPath = 'E:\\mladhd\\datasets\\text\\'

# Path to the tesseract executable
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Gonzalo\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = "E:\\Users\\ADHD Project\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

# Create the text dataset csv file
with open(os.path.join(textPath, f"{dataset}_focused2.csv"), "w", newline='', encoding='utf-8') as fp:
    wr = csv.writer(fp, delimiter=';')
    wr.writerow(["class", "text", "image"])

def process_image(image_path):
    img = cv2.imread(image_path)
    dilation = process_screenshot(img)
    contours = bounding_box(dilation)
    im2 = img.copy()
    text_list = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cropped = im2[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)
        text_list.append(text)
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
    for class_name in os.listdir(dataPath):
        # For each image in the class
        c = 0
        for image_name in os.listdir(os.path.join(dataPath, class_name)):
            start = time.time()
            c += 1
            #if c > limit_per_class:
            #    break
            print(f"Processing {c}: {image_name}")
            image_path = os.path.join(dataPath, class_name, image_name)
            text = process_image(image_path)
            with open(os.path.join(textPath, f"{dataset}_focused2.csv"), "a", newline='', encoding='utf-8') as fp:
                wr = csv.writer(fp, delimiter=';')
                if class_name == "0_focused":
                    wr.writerow([0, text, image_name])
                else:
                    wr.writerow([1, text, image_name])
            end = time.time()
            print(f"Time elapsed (s): {end - start}")