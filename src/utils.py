"""
License Plate Recognition Utilities

Author: Pratik Walunj
Description:
Helper class implementing the license plate detection
and recognition pipeline using OpenCV and EasyOCR.
"""

import cv2
import numpy as np
import easyocr
import imutils


class LicensePlateRecognizer:
    """
    License Plate Recognition pipeline using OpenCV and EasyOCR.

    Steps:
    1. Image preprocessing
    2. Plate contour detection
    3. Plate region extraction
    4. OCR-based text recognition
    5. Visualization of results
    """
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)
        return gray, edges

    def detect_plate(self, edges):
        keypoints = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                return approx

        return None

    def extract_plate(self, image, gray, contour):
        mask = np.zeros(gray.shape, np.uint8)

        cv2.drawContours(mask, [contour], 0, 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        cropped = gray[x1:x2 + 1, y1:y2 + 1]
        return cropped

    def recognize_text(self, plate_image):

        result = self.reader.readtext(plate_image)

        if len(result) > 0:
            return result[0][-2]

        return ""

    def draw_results(self, image, contour, text):

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            image,
            text,
            (contour[0][0][0], contour[1][0][1] + 60),
            font,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            image,
            tuple(contour[0][0]),
            tuple(contour[2][0]),
            (0, 255, 0),
            3,
        )

        return image