# License Plate Recognition (OpenCV + EasyOCR)

Real-time Automatic License Plate Recognition (ALPR) system built using OpenCV and EasyOCR. The system detects license plates from images or live camera feed, extracts the plate region using contour detection, and recognizes the plate number using optical character recognition.

---

# Overview

This project implements a lightweight license plate recognition pipeline using classical computer vision techniques and OCR. The system processes an image or real-time camera feed to locate the license plate and extract the plate text.

The pipeline performs:

- Image preprocessing
- Edge detection
- Contour-based plate localization
- Plate region extraction
- Text recognition using EasyOCR
- Visualization of detected plate and text


---

# Features

- License plate detection using contour analysis
- Optical Character Recognition using EasyOCR
- Real-time recognition from webcam
- Lightweight pipeline using classical computer vision
- Modular and maintainable code structure
- Easy to extend for embedded or edge deployment

---

# Installation

Clone the repository

```bash
git clone https://github.com/pratik7229/license-plate-recognition-opencv-easyocr.git
```
Create a virtual environment

```bash
python3 -m venv venv
```
Activate the virtual environment

```bash
source venv/bin/activate
```

Install dependencies (make sure you are in the root directory of project)

```bash
pip install -r requirements.txt
```
Running the project video pipeline

```bash
python3 video_lpr.py
```

Running the project image pipeline

```bash
python3 image_lpr.py
```


---

# Performance Considerations

For real-time performance on resource-constrained systems:

- Reduce frame resolution
- Initialize OCR model once
- Limit contour search
- Avoid expensive image filters

These optimizations improve processing speed significantly.

---

# Applications

This system can be used in:

- Smart parking systems
- Traffic monitoring
- Toll collection systems
- Vehicle access control
- Autonomous vehicle perception systems

---

# Future Improvements

Possible improvements include:

- Deep learning based plate detection
- Plate tracking across frames
- GPU acceleration using TensorRT
- Deployment on embedded platforms (Jetson / Raspberry Pi)
- Support for multiple license plate formats

---

# Author

Pratik Walunj  
Robotics Engineer | Robot Perception | Edge AI | Computer Vision

---

# License

This project is licensed under the Apache License 2.0 License.