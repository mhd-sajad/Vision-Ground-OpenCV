# 🎯 Vision Studio — Project Explained

> A Streamlit-based Computer Vision app powered by OpenCV, broken down **feature by feature**, **file by file**, **line by line**.

---

## 📁 Project Structure

```
Project OpencvApplications/
│
├── app.py                        ← Main Streamlit app (UI + routing)
├── models/                       ← Pre-trained model files
│   ├── MobileNetSSD_deploy.prototxt
│   ├── MobileNetSSD_deploy.caffemodel
│   ├── haarcascade_frontalface_default.xml
│   └── facial_expression_recognition_mobilefacenet_2022july.onnx
│
├── src/                          ← All feature logic lives here
│   ├── face_recognition.py
│   ├── object_detection.py
│   ├── document_enhancer.py
│   ├── edge_detection.py
│   ├── blur_detection.py
│   ├── ocr_scanner.py
│   ├── facial_emotion_detection.py
│   ├── gradients.py              ← (helper, not wired to UI yet)
│   ├── template_matching.py      ← (helper, not wired to UI yet)
│   └── utils.py                  ← Legacy image reader utility
│
└── .streamlit/config.toml        ← Streamlit theme config
```

---

## 🚀 app.py — The Main Application

The entry point of the entire project. It builds the UI, accepts image input, routes to the correct feature function, and shows the result.

### Imports & Setup (Lines 1–22)

```python
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from src.face_recognition import detect_faces
from src.object_detection import detect_objects
...
```

- `streamlit` → Builds the entire web UI (sidebar, buttons, file uploader etc.)
- `PIL.Image` → Opens uploaded image files
- `numpy` → Converts the PIL image into a pixel array for OpenCV
- `cv2` → OpenCV — the core computer vision engine
- Each `from src.X import Y` → Pulls in the individual feature function

```python
st.set_page_config(
    page_title='Vision Studio',
    layout="wide",
    initial_sidebar_state="expanded"
)
```
- Sets the browser tab title to `Vision Studio`
- `layout="wide"` → Makes the page use full screen width
- `initial_sidebar_state="expanded"` → Sidebar is open by default

---

### Custom CSS Dark Theme (Lines 24–168)

```python
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)
```

This injects raw CSS into the Streamlit app. Here's what each section does:

| CSS Target | What it Does |
|---|---|
| `@import` Google Fonts | Loads the **Plus Jakarta Sans** font |
| `.stApp` | Sets the main background to dark `#1a1b26` (Tokyo Night theme) |
| `section[data-testid="stSidebar"]` | Dark sidebar background `#16161e` with a faint purple border |
| `div.stButton > button` | Gradient purple→blue buttons with glow shadow |
| `[data-testid="stFileUploader"]` | Dashed purple border around upload zone |
| `div[role="radiogroup"] > label` | Each radio option is a styled box/card |
| `label:has(input:checked)` | The selected radio card glows with purple highlight |
| `.stImage` | Images get a card-like dark background with rounded corners |

---

### Gradient Header (Lines 170–200)

```python
st.markdown("""
    <div style="text-align: center;">
        <h1 style="background: linear-gradient(135deg, #bb9af3 0%, #7aa2f7 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;">
            VISION STUDIO
        </h1>
        <div>POWERED BY OPENCV</div>
        <p>Explore powerful computer vision filters...</p>
    </div>
""", unsafe_allow_html=True)
```

- Creates a big centered heading with a **purple-to-blue gradient text** effect
- `-webkit-background-clip: text` clips the gradient to the text shape only
- `-webkit-text-fill-color: transparent` makes the text itself transparent so the gradient shows through

---

### Sidebar Feature Selection (Lines 202–236)

```python
st.sidebar.markdown(...)   # Renders "OPENCV / VISION STUDIO" branding in sidebar

feature = st.sidebar.radio(
    "Choose Feature",
    ["Face Detection", "Object Detection", "Document Enhancer",
     "Edge Detection", "Blur Detection", "OCR Scanner", "Facial Emotion Detection"]
)
```

- `st.sidebar.radio(...)` creates a list of clickable radio options in the sidebar
- Whichever option the user clicks, `feature` holds its string name
- This string is used later to decide which function to call

---

### Input Method Selection (Lines 237–268)

```python
input_option = st.radio(
    "Choose Input Method",
    ["Upload Image", "Use Webcam"],
    horizontal=True
)
```

- Two options: Upload a file or use the webcam
- `horizontal=True` → shows them side-by-side

```python
if input_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)   # PIL opens it
        image = np.array(image)             # Convert to NumPy array (for OpenCV)
```

- `st.file_uploader` renders the drag-and-drop upload box
- `Image.open(uploaded_file)` → PIL reads the file
- `np.array(image)` → converts the image to a NumPy matrix of pixel values (height × width × 3 channels)

```python
elif input_option == "Use Webcam":
    camera_image = st.camera_input("Capture an Image")
    if camera_image is not None:
        image = Image.open(camera_image)
        image = np.array(image)
```

- `st.camera_input` opens the browser's webcam and lets the user snap a photo
- Same conversion pipeline applies after capture

---

### Processing & Display (Lines 270–405)

```python
if image is not None:
    col1, col2 = st.columns(2)   # Two side-by-side columns
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, width='stretch')
```

- Only runs if an image exists
- `st.columns(2)` creates a two-panel layout
- Left column always shows the original unmodified image

```python
output = None

if feature == "Face Detection":
    output = detect_faces(image)
elif feature == "Object Detection":
    output = detect_objects(image)
...
```

- `output` starts as `None`
- Based on the selected `feature`, the right function is called and the result is stored in `output`

```python
if output is not None:
    with col2:
        st.subheader("✨ Processed Output")
        st.image(output, width='stretch')
```

- If we got a result, it's shown in the right column

```python
if st.button("💾 Save Output"):
    save_image = output
    if len(output.shape) == 3:
        save_image = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite("outputs/result.jpg", save_image)
    st.success("✅ Output saved in outputs/result.jpg")
```

- The Save button converts RGB → BGR (because OpenCV saves in BGR)
- Writes the file as `outputs/result.jpg`
- Shows a success message

---

## 🟢 Feature 1 — Face Detection

**File:** `src/face_recognition.py`

### What it does
Uses OpenCV's built-in Haar Cascade classifier to detect human faces and draws green rectangles around them.

### Line by Line

```python
import cv2
```
Imports OpenCV.

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```
- `cv2.data.haarcascades` → the folder path where OpenCV stores its pre-trained XML classifiers
- `haarcascade_frontalface_default.xml` → The Haar Cascade model trained to recognize frontal faces
- This loads the model **once at import time** (not every call) for efficiency

```python
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```
- Converts the input RGB image to **grayscale** — Haar Cascade works on single-channel images
- `cv2.COLOR_RGB2GRAY` → the conversion code

```python
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
```
- `detectMultiScale` → scans the image at multiple scales to find faces of different sizes
- `scaleFactor=1.1` → shrinks the image by 10% each pass (the pyramid step)
- `minNeighbors=5` → a face candidate must be confirmed by 5 neighboring detections to be accepted (reduces false positives)
- Returns a list of rectangles: `[(x, y, w, h), ...]`

```python
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
```
- Loops over each detected face rectangle
- `(x, y)` → top-left corner, `(x+w, y+h)` → bottom-right corner
- `(0, 255, 0)` → Green color in BGR
- `2` → Line thickness of 2 pixels

```python
    return image
```
- Returns the original image with green boxes drawn on top of detected faces

---

## 🟠 Feature 2 — Object Detection

**File:** `src/object_detection.py`

### What it does
Uses a pre-trained **MobileNet-SSD** deep learning model to detect and label 20 different types of objects in an image.

### Line by Line

```python
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]
```
- A list of 21 class names (index 0 = background, indices 1–20 = real objects)
- The model's output returns a class index; we use this list to convert that index to a human-readable name

```python
net = cv2.dnn.readNetFromCaffe(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel"
)
```
- Loads the MobileNet-SSD model — a lightweight deep neural network designed for fast object detection
- `.prototxt` → defines the network **architecture** (layers, shapes)
- `.caffemodel` → contains the **learned weights** (what the network knows)
- Again, loaded **once at import** for speed

```python
def detect_objects(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
```
- Safety check: if someone passes a grayscale image, convert it to 3-channel BGR so the model works correctly

```python
    (h, w) = image.shape[:2]
```
- Grab image height and width — needed to scale detection boxes back to original size

```python
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )
```
- `cv2.resize(image, (300,300))` → MobileNet-SSD requires exactly 300×300 input
- `blobFromImage` → converts the image to a "blob" (a 4D tensor: batch × channels × height × width)
- `0.007843` → scale factor = `1 / 127.5` (normalizes pixel values from 0–255 to approximately -1 to +1)
- `127.5` → mean subtraction value (centers the pixel values)

```python
    net.setInput(blob)
    detections = net.forward()
```
- Feeds the blob into the network
- `net.forward()` → runs the forward pass through all layers → returns an array of detections

```python
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
```
- The output is a `(1, 1, N, 7)` array where N is the number of candidate detections
- Index `[0,0,i,2]` → the **confidence score** of the i-th detection

```python
        if confidence > 0.4:
```
- Only process detections with confidence above 40% — filters out weak guesses

```python
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = CLASSES[idx]
```
- `idx` → class index from detection output
- `detections[..., 3:7]` → normalized box coordinates (values between 0 and 1)
- Multiply by `[w, h, w, h]` → scale back to actual pixel coordinates
- `CLASSES[idx]` → convert class index to a label string

```python
            cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)
            cv2.putText(image, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
```
- Draws a green bounding box
- Draws the label in blue text 10 pixels above the box top

---

## 🔵 Feature 3 — Document Enhancer

**File:** `src/document_enhancer.py`

### What it does
Converts a photo of a document (or any image) into a clean, high-contrast black-and-white version — like a scanned document.

### Line by Line

```python
def enhance_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```
- Convert to grayscale — document enhancement works on intensity, not color

```python
    blur = cv2.GaussianBlur(gray, (5,5), 0)
```
- Apply a 5×5 Gaussian blur to **reduce noise** (camera grain, compression artifacts)
- This smooths the image before thresholding, so we get cleaner results

```python
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
```
- `adaptiveThreshold` → instead of using a single global threshold, it calculates a **local threshold** for each pixel based on its neighbors
- `255` → max output value (pixels become either 0 or 255)
- `cv2.ADAPTIVE_THRESH_GAUSSIAN_C` → weights neighbors by a Gaussian (closer pixels matter more)
- `cv2.THRESH_BINARY` → pixels above threshold → 255 (white), below → 0 (black)
- `11` → block size: the neighborhood area is 11×11 pixels
- `2` → constant subtracted from the weighted mean (fine-tunes the threshold)

```python
    return thresh
```
- Returns the crisp black-and-white document image

---

## 🟡 Feature 4 — Edge Detection

**File:** `src/edge_detection.py`

### What it does
Highlights the edges (boundaries between different regions) in an image using the **Canny Edge Detection** algorithm.

### Line by Line

```python
def detect_edges(image, threshold1=100, threshold2=200):
```
- Accepts two thresholds as parameters — in `app.py`, these come from sidebar sliders the user can adjust

```python
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```
- Canny works on grayscale images

```python
    edges = cv2.Canny(gray, threshold1, threshold2)
```
- **Canny Algorithm** runs in 4 steps internally:
  1. **Gaussian smoothing** → reduces noise
  2. **Gradient calculation** → finds intensity changes in x and y directions
  3. **Non-maximum suppression** → thins edges to 1-pixel width
  4. **Hysteresis thresholding** → keeps edges between `threshold1` and `threshold2`
- `threshold1` = lower threshold (edges weaker than this are discarded)
- `threshold2` = upper threshold (edges stronger than this are definitely kept)
- Edges in between are kept only if connected to a strong edge

```python
    return edges
```
- Returns a binary image: white pixels = edges, black pixels = background

### In app.py (Edge Detection controls)
```python
threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
output = detect_edges(image, threshold1, threshold2)
```
- Two sliders in the sidebar let the user interactively tune the edge sensitivity

---

## 🟣 Feature 5 — Blur Detection

**File:** `src/blur_detection.py`

### What it does
Measures how blurry an image is and classifies it as **BLURRY** or **SHARP** using the **Variance of Laplacian** method.

### Line by Line

```python
def detect_blur(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```
- Convert to grayscale — blur detection is purely intensity-based

```python
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
```
- The **Laplacian operator** is a second-order derivative filter
- It detects rapid intensity changes (edges, details)
- A sharp image has lots of rapid changes → high Laplacian values
- A blurry image has smooth gradients → low Laplacian values
- `cv2.CV_64F` → 64-bit float output to preserve negative values

```python
    variance = laplacian.var()
```
- Calculates the **variance** (spread of values) across all Laplacian pixel values
- High variance → image has lots of edges and detail → **SHARP**
- Low variance → image is smooth with few edges → **BLURRY**

```python
    is_blurry = variance < threshold
    return variance, is_blurry
```
- If variance is below 100 (default) → classified as blurry
- Returns both the score and the boolean classification

### In app.py (Blur Detection display)
```python
variance, is_blurry = detect_blur(image)
status = "BLURRY" if is_blurry else "SHARP"
output = image.copy()
cv2.putText(output, f"Blur Score: {variance:.2f}", (10, 30), ...)
cv2.putText(output, f"Status: {status}", (10, 60), ...)
```
- The score and status are shown both in the sidebar AND overlaid on the image with `cv2.putText`

---

## 🔴 Feature 6 — OCR Scanner

**File:** `src/ocr_scanner.py`

### What it does
Reads text from an image using **Tesseract OCR** and highlights where each word was found with green bounding boxes.

### Line by Line

```python
import pytesseract
```
- Python wrapper for **Tesseract OCR** — Google's open-source Optical Character Recognition engine

```python
def configure_tesseract():
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        ...
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
```
- Tesseract is an external program; this function tries common Windows installation paths to auto-detect it
- If none found, tries to call it via PATH (works on Linux/Mac where it's installed system-wide)

```python
TESSERACT_CONFIGURED = configure_tesseract()
```
- Runs the detection **at import time** and stores the result in a boolean

```python
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
```
- Converts to grayscale
- Blurs to reduce noise
- `cv2.THRESH_OTSU` → automatically finds the optimal global threshold value (Otsu's method)
- `cv2.THRESH_BINARY + cv2.THRESH_OTSU` → combines binarization with auto-threshold selection
- Result: Clean black-and-white image that Tesseract reads much more accurately

```python
def perform_ocr(image):
    processed = preprocess_for_ocr(image)
    text = pytesseract.image_to_string(processed)
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    return text, data, processed
```
- `image_to_string` → runs OCR and returns all recognized text as a single string
- `image_to_data` → returns detailed info per word: position (x, y, w, h), confidence score, and text
- `output_type=pytesseract.Output.DICT` → returns this data as a Python dictionary

```python
def draw_text_boxes(image, data, confidence_threshold=60):
    img_copy = image.copy()
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > confidence_threshold:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,255,0), 2)
    return img_copy
```
- For each detected word, only draws a box if Tesseract's confidence is above 60%
- Filters out uncertain/noise detections
- Green boxes highlight where text was found

### In app.py (OCR section)
```python
text, data, processed = perform_ocr(image)
result_with_boxes = draw_text_boxes(image, data)
st.sidebar.text_area("Extracted Text", text, height=200)
output = result_with_boxes
```
- Extracted text appears in the sidebar text area
- Image with green word boxes shown as the processed output

---

## 🟤 Feature 7 — Facial Emotion Detection

**File:** `src/facial_emotion_detection.py`

### What it does
Detects faces in an image and then predicts the **emotion** on each face using a pre-trained **MobileFaceNet ONNX model** from OpenCV Zoo. Falls back to a brightness-based heuristic if the model is unavailable.

### Line by Line

```python
EMOTIONS_ONNX = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
```
- The 7 emotion classes the ONNX model can predict, in the order the model's output neurons correspond to

```python
class EmotionDetector:
    def __init__(self):
        self.net = None
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models',
                      'facial_expression_recognition_mobilefacenet_2022july.onnx')
        if os.path.exists(model_path):
            self.net = cv2.dnn.readNetFromONNX(model_path)
```
- The class loads the ONNX model once on instantiation
- `os.path.dirname(__file__)` → gets the `src/` folder path, then navigates to `models/`
- `cv2.dnn.readNetFromONNX` → OpenCV's built-in ONNX loader

```python
    def predict(self, face_rgb):
        face_resized = cv2.resize(face_rgb, (112, 112))
```
- MobileFaceNet expects exactly **112×112** input

```python
        face_float = face_resized.astype(np.float32) / 255.0
        face_float = (face_float - 0.5) / 0.5
```
- Scale pixels from `[0, 255]` to `[0, 1]`
- Then normalize to `[-1, 1]` using `(x - 0.5) / 0.5` — this is what OpenCV Zoo's model was trained with

```python
        blob = cv2.dnn.blobFromImage(face_float)
        self.net.setInput(blob, 'data')
        preds = self.net.forward(['label'])
```
- Creates a blob and feeds it to the network
- `'data'` and `'label'` are the specific layer names defined in this model's ONNX architecture

```python
        logits = preds[0]
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / np.sum(exp_preds)
```
- The model outputs **logits** (raw scores), not probabilities
- Apply **softmax** manually: subtract max for numerical stability, then divide by sum → converts to probabilities that sum to 1

```python
        class_idx = np.argmax(probs[0])
        emotion = EMOTIONS_ONNX[class_idx]
        confidence = float(probs[0][class_idx])
        return emotion, confidence
```
- `argmax` finds the index with the highest probability
- Maps that index to the emotion label
- Returns the emotion name and its confidence score

---

```python
detector = EmotionDetector()
```
- **Module-level instantiation** — the model is loaded once when the file is imported, not on every prediction call

---

```python
def detect_emotions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
```
- First detects faces using **Haar Cascade** (same technique as Face Detection feature)
- `minSize=(30,30)` → ignore face detections smaller than 30×30 pixels (too small to analyse emotion)

```python
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        prediction = detector.predict(face_roi)
```
- Crops out each detected face from the image (`image[y:y+h, x:x+w]` — NumPy slice)
- Sends the cropped face to the ONNX model for emotion prediction

```python
        if prediction is not None:
            emotion, confidence = prediction
        else:
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi_gray, (48, 48))
            face_roi_normalized = face_roi_resized / 255.0
            emotion, confidence = predict_emotion_simple(face_roi_normalized)
```
- If the ONNX model fails or isn't loaded, fall back to `predict_emotion_simple`

```python
def predict_emotion_simple(face_roi):
    mean_intensity = np.mean(face_roi)
    std_intensity = np.std(face_roi)

    if std_intensity > 0.15:        # High contrast image
        if mean_intensity > 0.6:    # Bright → Surprise
            return 'Surprise', 0.75
        else:                       # Dark → Angry
            return 'Angry', 0.70
    elif mean_intensity > 0.7:      # Very bright → Happy
        return 'Happy', 0.80
    elif mean_intensity < 0.3:      # Dark → Sad
        return 'Sad', 0.65
    else:
        return 'Neutral', 0.60
```
- A pure math-based fallback with no ML
- Uses **mean pixel intensity** (brightness) and **standard deviation** (contrast) as crude emotion proxies
- Not accurate for real use but prevents the app from crashing if the model is missing

---

## 🔧 Helper Files (Not in UI Yet)

### `src/gradients.py`
Computes **Sobel X, Sobel Y, and Laplacian** gradient maps from an image — useful for visualizing edges in different directions.

```python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)   # Horizontal edges
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)   # Vertical edges
laplacian = cv2.Laplacian(gray, cv2.CV_64F)   # All-direction edges
```

### `src/template_matching.py`
Finds where a **template image** appears inside a larger image using `cv2.matchTemplate`.

```python
result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
locations = zip(*((result >= 0.8).nonzero()[::-1]))
```
- `TM_CCOEFF_NORMED` → normalized cross-correlation scoring (1.0 = perfect match)
- Threshold of 0.8 → only draw boxes where similarity is >= 80%

### `src/utils.py`
A legacy utility for reading images from Streamlit's UploadedFile object using raw bytes.

```python
def read_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)      # Decode to BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return image
```
- Not currently used in `app.py` (replaced by `PIL.Image.open` + `np.array`), but kept as a reference

---

## 🤖 Models Directory

| File | Purpose |
|---|---|
| `haarcascade_frontalface_default.xml` | OpenCV's pre-trained Haar Cascade for frontal face detection |
| `MobileNetSSD_deploy.prototxt` | MobileNet-SSD network architecture definition |
| `MobileNetSSD_deploy.caffemodel` | MobileNet-SSD trained weights (trained on PASCAL VOC 2007+2012) |
| `facial_expression_recognition_mobilefacenet_2022july.onnx` | MobileFaceNet emotion classifier from OpenCV Zoo (July 2022) |

---

## 🌊 How Data Flows Through the App

```
User uploads image
        ↓
PIL.Image.open() → np.array()   [converts file to pixel matrix]
        ↓
app.py reads selected feature from sidebar
        ↓
Calls matching src/ function with the image array
        ↓
src/ function processes image using OpenCV
        ↓
Returns processed image (or tuple with extra data)
        ↓
app.py displays original + processed side by side
        ↓
User optionally saves the output
```

---

*Barakallahu feek! May this documentation make your learning journey clear and blessed.*
