# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import os

# Emotion labels for MobileFaceNet in OpenCV Zoo
EMOTIONS_ONNX = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# Fallback heuristic labels
EMOTIONS_FALLBACK = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionDetector:
    def __init__(self):
        self.net = None
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'facial_expression_recognition_mobilefacenet_2022july.onnx')
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
            except Exception as e:
                print(f"Error loading ONNX model: {str(e)}")
        else:
            print(f"Warning: ONNX model not found at {model_path}")

    def predict(self, face_rgb):
        if self.net is None:
            # Fallback if model not loaded
            return None
        
        try:
            # MobileFaceNet model preprocessing:
            # 1. Resize to 112x112
            face_resized = cv2.resize(face_rgb, (112, 112))
            
            # 2. Convert to float32 and scale to [0, 1]
            face_float = face_resized.astype(np.float32, copy=False) / 255.0
            
            # 3. Standardize: (image - mean) / std (using 0.5 for both as per OpenCV Zoo model config)
            face_float = (face_float - 0.5) / 0.5
            
            # 4. Create blob: shape (1, 3, 112, 112)
            blob = cv2.dnn.blobFromImage(face_float)
            
            self.net.setInput(blob, 'data')
            preds = self.net.forward(['label'])
            
            # 5. Postprocess with softmax to get probabilities
            logits = preds[0]
            exp_preds = np.exp(logits - np.max(logits))
            probs = exp_preds / np.sum(exp_preds)
            
            class_idx = np.argmax(probs[0])
            emotion = EMOTIONS_ONNX[class_idx]
            confidence = float(probs[0][class_idx])
            
            return emotion, confidence
        except Exception as e:
            print(f"ONNX Inference Error: {str(e)}")
            return None

# Instantiate detector once at module load
detector = EmotionDetector()

def detect_emotions(image):
    """
    Detect emotions in faces within an image.

    Args:
        image: Input image (RGB format)

    Returns:
        tuple: (image_with_emotions, emotion_results)
               image_with_emotions: Image with emotion labels drawn
               emotion_results: List of detected emotions with confidence
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Load face cascade
    face_cascade_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'haarcascade_frontalface_default.xml')
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"Face cascade not found at {face_cascade_path}")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    image_with_emotions = image.copy()
    emotion_results = []

    for (x, y, w, h) in faces:
        # Extract face ROI in RGB (since detector expects RGB)
        face_roi = image[y:y+h, x:x+w]

        # Attempt to predict emotion using pre-trained ONNX model
        prediction = detector.predict(face_roi)

        if prediction is not None:
            emotion, confidence = prediction
        else:
            # Fallback to simple heuristic prediction
            # Convert crop to gray and resize to 48x48 as expected by simple heuristic
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi_gray, (48, 48))
            face_roi_normalized = face_roi_resized / 255.0
            emotion, confidence = predict_emotion_simple(face_roi_normalized)

        # Draw rectangle around face
        cv2.rectangle(image_with_emotions, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw emotion label
        label = f"{emotion}: {confidence:.2f}"
        cv2.putText(image_with_emotions, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        emotion_results.append({
            'emotion': emotion,
            'confidence': confidence,
            'bbox': (x, y, w, h)
        })

    return image_with_emotions, emotion_results

def predict_emotion_simple(face_roi):
    """
    Simple emotion prediction based on basic image features (grayscale normalized 48x48).
    This serves as a fallback.

    Args:
        face_roi: Normalized face region (48x48)

    Returns:
        tuple: (predicted_emotion, confidence)
    """
    # Calculate basic stats
    mean_intensity = np.mean(face_roi)
    std_intensity = np.std(face_roi)

    if std_intensity > 0.15:  # High contrast
        if mean_intensity > 0.6:  # Bright
            return 'Surprise', 0.75
        else:  # Dark
            return 'Angry', 0.70
    elif mean_intensity > 0.7:  # Very bright
        return 'Happy', 0.80
    elif mean_intensity < 0.3:  # Dark
        return 'Sad', 0.65
    else:
        return 'Neutral', 0.60

def get_available_emotions():
    """Return list of detectable emotions."""
    return EMOTIONS_ONNX.copy()