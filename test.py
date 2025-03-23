import cv2

def find_cameras():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available")
            cap.release()
        else:
            print(f"Camera {i} not found")

find_cameras()
