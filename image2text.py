import os
import pytesseract
import cv2
import csv
from feature_logger import process_screenshot, bounding_box, OCR

# Path to the folder containing the images
imagePath = 'D:\\Alerta Backup Data\\gonzalo_data\\datsets\\image\\sample_dataset\\'

# Path to the folder that will contain the text dataset
textPath = 'D:\\Alerta Backup Data\\gonzalo_data\\datsets\\text\\'

# Path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Gonzalo\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

# For each class in the dataset
for class_name in os.listdir(imagePath):
    # For each image in the class
    for image_name in os.listdir(os.path.join(imagePath, class_name)):
        # Read the image
        img = cv2.imread(os.path.join(imagePath, class_name, image_name))
        # Process the image
        img = process_screenshot(img)
        # Get the bounding boxes
        boxes = bounding_box(img)
        # Extract the text from the image
        text = OCR(img, boxes)
        # Write the text to the csv file
        with open(os.path.join(textPath, "text_dataset.csv"), "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow([image_name, text, class_name])