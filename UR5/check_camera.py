import cv2
cap = cv2.VideoCapture(1)

# Try to disable autofocus
result = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
print(f"Autofocus control supported: {result}")  # True = camera accepted the command

# Read back what the camera reports
af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
print(f"Autofocus value: {af}")  # 0 = off, 1 = on, -1 = not supported

cap.release()