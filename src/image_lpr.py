import cv2
import matplotlib.pyplot as plt
from utils import LicensePlateRecognizer


def main():

    image = cv2.imread("/Users/pratik/Documents/Finalized Projects/Liscense_Plate_Recognition/data/image3.jpg")

    lpr = LicensePlateRecognizer()

    gray, edges = lpr.preprocess(image)

    contour = lpr.detect_plate(edges)

    if contour is None:
        print("License plate not detected")
        return

    plate = lpr.extract_plate(image, gray, contour)

    text = lpr.recognize_text(plate)

    result = lpr.draw_results(image, contour, text)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()