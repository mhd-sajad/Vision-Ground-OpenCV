import cv2
import numpy as np 

CLASSES = [
    "background", "aeroplane", "bicycle",
    "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

#Load Model
net = cv2.dnn.readNetFromCaffe(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel"
)

def detect_objects(image):
    # Ensure image is in color format (3 channels)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    (h,w) = image.shape[:2]

    blob=cv2.dnn.blobFromImage(
        cv2.resize(image,(300,300)),
        0.007843,
        (300, 300),
        127.5
    )
    net.setInput(blob)
    detections=net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.4:
            idx = int(detections[0,0,i,1])
            box = detections[0,0,i,3:7]*np.array(
                [w,h,w,h]
            )

            (startX,startY,endX,endY)=box.astype("int")
            label = CLASSES[idx]

            cv2.rectangle(
                image,
                (startX,startY),
                (endX,endY),
                (0,255,0),
                2
            )

            cv2.putText(
                image,
                label,
                (startX,startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,0,0),
                2
            )
    return image