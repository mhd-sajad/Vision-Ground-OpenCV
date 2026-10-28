import cv2
import pytesseract
import numpy as np
import os

# Configure Tesseract path for Windows
def configure_tesseract():
    """Configure pytesseract to work with Tesseract OCR."""
    # MANUAL OVERRIDE: Uncomment and set your Tesseract path here if auto-detection fails
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Common Tesseract installation paths on Windows
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\anupa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        r'D:\Tesseract-OCR\tesseract.exe',
        r'C:\Users\anupa\miniconda3\envs\cvenv\Library\bin\tesseract.exe',  # conda path
        r'C:\Users\anupa\miniconda3\Library\bin\tesseract.exe'  # another conda path
    ]

    # Try to find Tesseract in common locations
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"Tesseract configured at: {path}")
            return True

    # If not found, try to use it from PATH
    try:
        pytesseract.get_tesseract_version()
        print("Tesseract found in PATH")
        return True
    except pytesseract.TesseractNotFoundError:
        print("Tesseract not found in any location")
        return False

# Configure Tesseract on import
TESSERACT_CONFIGURED = configure_tesseract()

def preprocess_for_ocr(image):
    """Preprocess image for better OCR results."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def perform_ocr(image):
    """
    Perform OCR on an image.

    Args:
        image: Input image (RGB)

    Returns:
        tuple: (extracted_text, ocr_data_dict, processed_image)
    """
    if not TESSERACT_CONFIGURED:
        raise Exception(
            "Tesseract OCR is not installed or not found in PATH.\n\n"
            "QUICK INSTALLATION:\n"
            "1. Download from: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.4.20240514.exe\n"
            "2. Run the installer\n"
            "3. Check 'Add to PATH' during installation\n"
            "4. Restart your application\n\n"
            "ALTERNATIVE: conda install -c conda-forge tesseract\n\n"
            "The code will automatically detect Tesseract in common locations."
        )

    # Preprocess the image
    processed = preprocess_for_ocr(image)

    # Extract text using pytesseract
    text = pytesseract.image_to_string(processed)

    # Get detailed data including bounding boxes
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

    return text, data, processed

def draw_text_boxes(image, data, confidence_threshold=60):
    img_copy = image.copy()
    n_boxes = len(data['text'])

    for i in range(n_boxes):
        if int(data['conf'][i]) > confidence_threshold:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_copy