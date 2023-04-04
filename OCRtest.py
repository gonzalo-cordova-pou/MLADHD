import cv2
import numpy as np
import pytesseract

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Gonzalo\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

for num in range(1, 16):

    print("Processing image {}".format(num))

    # Read image from which text needs to be extracted
    img = cv2.imread("C:\\Users\\Gonzalo\\Documents\\MLADHD\\test_images\\{}.jpg".format(num))

    # Crop 20px from each side of the image, except 60px from the bottom
    img = img[0:img.shape[0] - 60, 20:img.shape[1] - 20]
    
    # Preprocessing the image starts
    
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
    
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)

    # Sort all the contours avoiding annotations
    # by x-coordinate (left to right), then y-coordinate (top to bottom), then area (big to small)
    lambda_sort = lambda x: (cv2.boundingRect(x)[0], cv2.boundingRect(x)[1], cv2.contourArea(x))
    contours = sorted(contours, key=lambda_sort)
    
    # Creating a copy of image
    im2 = img.copy()
    
    # A text file is created and flushed
    file = open(".//text_output//OCR_{}.txt".format(num), "w+")
    file.write("")
    file.close()
    
    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # If the area of the rectangle is more than 80% of the image area, then ignore it
        if (w * h) / (img.shape[0] * img.shape[1]) < 0.95:

            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]

            # Add the coordinates of the bounding box to the file
            coord = "x: " + str(x) + ", y: " + str(y) + ", w: " + str(w) + ", h: " + str(h) + "\n"

            # Apply OCR on the cropped image
            text = pytesseract.image_to_string(cropped)

            if len(text) > 3:
                # Drawing a rectangle on copied image
                rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (36, 255, 12), 2)

                # Open the file in append mode
                file = open(".//text_output//OCR_{}.txt".format(num), "a")

                # Appending the text into file
                file.write(coord)
                file.write(text)
                file.write("\n")
                
                # Close the file
                file.close

    # Save a version of the image with the bounding boxes
    cv2.imwrite("C:\\Users\\Gonzalo\\Documents\\MLADHD\\text_output\\boxes_{}.jpg".format(num), im2)