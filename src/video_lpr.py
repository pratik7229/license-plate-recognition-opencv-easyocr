import cv2
from utils import LicensePlateRecognizer


def main():
    """
    Real-time license plate recognition using webcam feed.
    Press 'q' to exit the application.
    """
    cap = cv2.VideoCapture(0)
    lpr = LicensePlateRecognizer()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray, edges = lpr.preprocess(frame)
        contour = lpr.detect_plate(edges)
        
        if contour is not None:
            plate = lpr.extract_plate(frame, gray, contour)
            text = lpr.recognize_text(plate)
            frame = lpr.draw_results(frame, contour, text)

        cv2.imshow("License Plate Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()