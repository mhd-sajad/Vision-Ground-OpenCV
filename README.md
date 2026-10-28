#  Vision Studio — OpenCV Suite

A modern, full-featured Computer Vision playground and toolkit built with **Python**, **OpenCV**, and **Streamlit**. Vision Studio brings classical computer vision pipelines and deep learning models together under an interactive, sleek UI.

---

## ✨ Features

- **👤 Face Recognition / Detection**: Detect human faces with adjustable scaling factors and minimum neighbor thresholds using Haar Cascades.
- **🎯 Object Detection**: Detect everyday objects with confidence score thresholds utilizing pre-trained MobileNet-SSD.
- **📄 Document Enhancer**: Clean, de-noise, and enhance lighting/contrast of scanned documents using adaptive thresholding and bilateral filtering.
- **⚡ Edge Detection**: Interactive multi-threshold edge extraction using the Canny algorithm.
- **🔍 Blur Detection**: Detect image sharpness and focal blur via the Laplacian variance operator.
- **📝 OCR Scanner**: Extract text and detect bounding word boxes with optical character recognition powered by Tesseract OCR.
- **😊 Facial Emotion Detection**: Recognize facial emotions (Happy, Sad, Angry, Surprised, Neutral, etc.) with pre-trained MobileFaceNet ONNX deep learning models.

---

## 📁 Repository Structure

```text
Vision-Studio-OpenCV/
├── app.py                      # Main Streamlit application frontend
├── requirements.txt            # Python dependencies
├── PROJECT_EXPLAINED.md        # In-depth architectural & mathematical explanations
├── models/                     # Deep learning weights and Haar cascades
│   ├── MobileNetSSD_deploy.caffemodel
│   ├── MobileNetSSD_deploy.prototxt
│   ├── haarcascade_frontalface_default.xml
│   └── facial_expression_recognition_mobilefacenet_2022july.onnx
└── src/                        # Modular processing pipelines
    ├── face_recognition.py
    ├── object_detection.py
    ├── document_enhancer.py
    ├── edge_detection.py
    ├── blur_detection.py
    ├── ocr_scanner.py
    ├── facial_emotion_detection.py
    └── utils.py
```

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/mhd-sajad/Vision-Studio-OpenCV.git
cd Vision-Studio-OpenCV
```

### 2. Set up virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on OCR**: Tesseract engine must be installed on your operating system:
> - **Ubuntu / Debian**: `sudo apt install tesseract-ocr`
> - **Arch Linux**: `sudo pacman -S tesseract`
> - **macOS**: `brew install tesseract`

### 4. Launch the application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Vision Core**: OpenCV (`cv2`), NumPy, Pillow
- **Deep Learning**: Caffe & ONNX Runtime (via OpenCV DNN)
- **OCR Engine**: Tesseract OCR & Pytesseract
- **UI Framework**: Streamlit

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

