import cv2 

#load Haar Cascade 
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

def detect_faces(image):
    #convert RGB to Gray
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    #Draw rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(
            image,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )
    
    return image