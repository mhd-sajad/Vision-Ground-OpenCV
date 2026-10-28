# pyrefly: ignore [missing-import]
import streamlit as st 
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import numpy as np 
# pyrefly: ignore [missing-import]
import cv2 
from src.face_recognition import detect_faces
from src.object_detection import detect_objects
from src.document_enhancer import enhance_document
from src.edge_detection import detect_edges
from src.blur_detection import detect_blur
from src.ocr_scanner import perform_ocr, draw_text_boxes
from src.facial_emotion_detection import detect_emotions


st.set_page_config(
    page_title='Vision Studio',
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetic dark theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply font safely without breaking built-in icons */
    html, body, .stApp, .stMarkdown, p, label, input, select, textarea {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Main background and base colors */
    .stApp {
        background-color: #1a1b26;
        color: #a9b1d6;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #16161e !important;
        border-right: 1px solid rgba(187, 154, 243, 0.1);
    }
    
    /* Custom buttons style (gradient and shadow) */
    div.stButton > button {
        background: linear-gradient(135deg, #bb9af3 0%, #7aa2f7 100%) !important;
        color: #16161e !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.8rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(187, 154, 243, 0.3) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(187, 154, 243, 0.5) !important;
        color: #16161e !important;
    }
    
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* File uploader container */
    [data-testid="stFileUploader"] {
        background-color: #16161e !important;
        border: 2px dashed rgba(187, 154, 243, 0.3) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #7aa2f7 !important;
        background-color: #1e1e2d !important;
    }
    
    /* Card-like borders for output components and images */
    .stImage, [data-testid="stMetric"] {
        background-color: #16161e;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(187, 154, 243, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Slider color tuning to match theme accent */
    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #bb9af3 !important;
    }
    .stSlider [aria-valuenow] {
        background-color: #bb9af3 !important;
        border: 2px solid #16161e !important;
    }
    
    /* Input and selector formatting */
    input, textarea, select {
        background-color: #16161e !important;
        color: #a9b1d6 !important;
        border: 1px solid rgba(187, 154, 243, 0.15) !important;
        border-radius: 8px !important;
    }
    
    /* Style each individual radio option as a box */
    div[role="radiogroup"] > label {
        background-color: #1a1b26;
        padding: 12px 15px;
        border-radius: 10px;
        border: 2px solid rgba(187, 154, 243, 0.1);
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        display: flex;
        align-items: center;
        box-sizing: border-box;
    }
    
    /* Enforce equal width and height strictly for sidebar features */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        width: 100%;
        min-height: 55px;
    }
    
    /* Hover effect for radio options */
    div[role="radiogroup"] > label:hover {
        border-color: rgba(187, 154, 243, 0.4);
        background-color: rgba(187, 154, 243, 0.05);
        transform: translateY(-1px);
    }

    /* Highlight chosen one using CSS :has selector (supported in modern browsers) */
    div[role="radiogroup"] > label:has(input:checked) {
        border-color: #bb9af3;
        background-color: rgba(187, 154, 243, 0.15);
        box-shadow: 0 4px 15px rgba(187, 154, 243, 0.2);
    }
    
    /* Standard divider styling */
    hr {
        border-color: rgba(187, 154, 243, 0.1) !important;
    }
    
    /* Sidebar text/headers */
    section[data-testid="stSidebar"] label {
        color: #bb9af3 !important;
        font-weight: 600 !important;
    }

    /* Make header transparent so it doesn't visually block content */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Reduce top padding for main content */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }

    /* Pull sidebar content to the top */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Aesthetic Gradient Header
st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 2rem 0;">
        <h1 style="
            background: linear-gradient(135deg, #bb9af3 0%, #7aa2f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
            font-size: 3.8rem;
            margin-bottom: 0.2rem;
            letter-spacing: -2px;
            text-transform: uppercase;
        ">VISION STUDIO</h1>
        <div style="
            text-transform: uppercase;
            letter-spacing: 3px;
            color: #7aa2f7;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 1rem;
            font-family: 'Plus Jakarta Sans', sans-serif;
        ">POWERED BY OPENCV</div>
        <p style="
            color: #a9b1d6;
            font-size: 1.1rem;
            opacity: 0.8;
            font-weight: 400;
        ">Explore powerful computer vision filters and algorithms in real-time</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style= text-align: center;">
        <div style="
            text-transform: uppercase;
            letter-spacing: 3px;
            color: #7aa2f7;
            font-size: 0.85rem;
            font-weight: 700;
            font-family: 'Plus Jakarta Sans', sans-serif;
        ">OPENCV</div>  
        <h2 style="
            color: #bb9af3;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
            margin: 0;
            font-size: 1.8rem;
            line-height: 1.2;
            text-transform: uppercase;
        ">VISION STUDIO</h2>
    </div>
""", unsafe_allow_html=True)

#SIDEBAR
feature = st.sidebar.radio(
    "Choose Feature",
    [
        "Face Detection",
        "Object Detection",
        "Document Enhancer",
        "Edge Detection",
        "Blur Detection",
        "OCR Scanner",
        "Facial Emotion Detection"
    ]
)
# IMAGE INPUT OPTIONS
input_option = st.radio(
    "Choose Input Method",
    ["Upload Image", "Use Webcam"],
    horizontal=True
)

image = None

#IMAGE UPLOAD
if input_option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image = np.array(image)
#webcam input
elif input_option == "Use Webcam":

    camera_image = st.camera_input("Capture an Image")

    if camera_image is not None:

        image = Image.open(camera_image)

        image = np.array(image)

#process image
if image is not None:

    # Create columns
    col1, col2 = st.columns(2)

    # Show original image
    with col1:

        st.subheader("📷 Original Image")

        st.image(
            image,
            width='stretch'
        )
    
    #feature processing
    output = None

    # FACE DETECTION
    if feature == "Face Detection":

        output = detect_faces(image)

    # OBJECT DETECTION
    elif feature == "Object Detection":

        output = detect_objects(image)
    
    # DOCUMENT ENHANCER
    elif feature == "Document Enhancer":

        output = enhance_document(image)

    # EDGE DETECTION
    elif feature == "Edge Detection":

        threshold1 = st.sidebar.slider(
            "Threshold 1",
            0,
            255,
            100
        )
        threshold2 = st.sidebar.slider(
            "Threshold 2",
            0,
            255,
            200
        )
        output = detect_edges(
            image,
            threshold1,
            threshold2
        )

    # BLUR DETECTION
    elif feature == "Blur Detection":

        variance, is_blurry = detect_blur(image)
        status = "BLURRY" if is_blurry else "SHARP"
        st.sidebar.write(f"Blur Score: {variance:.2f}")
        st.sidebar.write(f"Status: {status}")

        # For display, we'll show the original image with blur status
        output = image.copy()
        cv2.putText(output, f"Blur Score: {variance:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(output, f"Status: {status}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # OCR SCANNER
    elif feature == "OCR Scanner":

        try:
            text, data, processed = perform_ocr(image)
            result_with_boxes = draw_text_boxes(image, data)

            st.sidebar.text_area("Extracted Text", text, height=200)

            output = result_with_boxes
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            st.info("💡 **Quick Fix:**\n"
                   "1. Download: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.4.20240514.exe\n"
                   "2. Install and check 'Add to PATH'\n"
                   "3. Restart the app\n\n"
                   "**Alternative:** Run `conda install -c conda-forge tesseract` in terminal")
            output = image  # Show original image as fallback

    # FACIAL EMOTION DETECTION
    elif feature == "Facial Emotion Detection":

        try:
            result_image, emotion_results = detect_emotions(image)

            # Display emotion results in sidebar
            if emotion_results:
                st.sidebar.write("### Detected Emotions:")
                for i, result in enumerate(emotion_results):
                    st.sidebar.write(f"**Face {i+1}:** {result['emotion']} ({result['confidence']:.2f})")
            else:
                st.sidebar.write("No faces detected in the image.")

            output = result_image
        except Exception as e:
            st.error(f"Emotion Detection Error: {str(e)}")
            output = image  # Show original image as fallback
    if output is not None:
        with col2:
            st.subheader("✨ Processed Output")

            st.image(
                output,
                width='stretch'
            )

        
        # SAVE OUTPUT BUTTON
        if st.button("💾 Save Output"):

            # Convert RGB -> BGR before saving
            save_image = output

            if len(output.shape) == 3:

                save_image = cv2.cvtColor(
                    output,
                    cv2.COLOR_RGB2BGR
                )

            cv2.imwrite(
                "outputs/result.jpg",
                save_image
            )

            st.success("✅ Output saved in outputs/result.jpg")