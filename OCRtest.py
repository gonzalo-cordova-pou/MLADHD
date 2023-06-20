import cv2
import os
import numpy as np
import pytesseract

IMAGE_PATH = ".\\sample_dataset\\"
IMAGE_PATH = ".\\results\\"

# Reference the installed location of Tesseract-OCR in your system
# Get Tesseract-OCR from: https://github.com/tesseract-ocr/tesseract
pytesseract.pytesseract.tesseract_cmd = "tesseract.exe"

# For each folder in the dataset
for folder in os.listdir(IMAGE_PATH):
    # For each image in the folder
    for num, file in enumerate(os.listdir(IMAGE_PATH + folder)):

        print(F"Processing image: {file}")

        # Read image from which text needs to be extracted
        img = cv2.imread(os.path.join(IMAGE_PATH, folder, file))

        # Crop 20px from each side of the image, except 60px from the bottom
        img = img[0:img.shape[0] - 60, 20:img.shape[1] - 20]
        
        # Preprocessing the image starts
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        # Performing OTSU threshold
        ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

        # Define the shape and size of the structure and kernel.
        # The kernel size determines the area of the rectangle used for detection.
        # Using a smaller value such as (10, 10) will detect individual words
        # rather than whole sentences.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 2)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)

        # Sort all the contours avoiding annotations
        # by x-coordinate (left to right), then y-coordinate (top to bottom),
        # then area (big to small)
        def lambda_sort(x):
            return (
                cv2.boundingRect(x)[0],
                cv2.boundingRect(x)[1],
                cv2.contourArea(x)
            )
        contours = sorted(contours, key=lambda_sort)
        im2 = img.copy()

        # A text file is created and flushed
        file = open(IMAGE_PATH + f"OCR_{folder}_{num}.txt", "w")
        file.write("")
        file.close()

        # Looping through the identified contours
        # Then rectangular part is cropped and passed on
        # to pytesseract for extracting text from it
        # Extracted text is then written into the text file
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (w * h) / (img.shape[0] * img.shape[1]) <= 1:

                # Cropping the text block for giving input to OCR
                cropped = im2[y:y + h, x:x + w]

                # Add the coordinates of the bounding box to the file
                coord = "x: " + str(x) + ", y: " + str(y) + ", w: " + str(w) + ", h: " + str(h) + "\n"

                # Apply OCR on the cropped image
                text = pytesseract.image_to_string(cropped)

                if len(text) > 3:
                    # Drawing a rectangle on copied image
                    rect = cv2.rectangle(
                        im2, (x, y),
                        (x + w, y + h),
                        (36, 255, 12),
                        2
                    )
                    # Open the file in append mode
                    file = open(IMAGE_PATH + f"OCR_{folder}_{num}.txt", "a")
                    # Appending the text into file
                    file.write(text)
                    # Close the file
                    file.close

            # Save a version of the image with the bounding boxes
            cv2.imwrite(IMAGE_PATH + f"OCR_{folder}_{num}.png", im2)
