import cv2

print("OpenCV is working!")
print("Version:", cv2.__version__)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Camera opened successfully!")

cap.release()