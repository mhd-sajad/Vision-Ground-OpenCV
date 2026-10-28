import cv2
import numpy as np 

def read_image(uploaded_file):
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes,1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    return image