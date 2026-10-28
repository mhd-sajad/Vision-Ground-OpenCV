import cv2

def detect_edges(image,threshold1=100,threshold2=200):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        gray,
        threshold1,
        threshold2
    )
    return edges
