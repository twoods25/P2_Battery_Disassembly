"""
hsv_tuner.py
------------
Live HSV threshold tuner. Point at your work area, adjust sliders
until the target color is clean white on black, then read off the values.

Usage:
    python hsv_tuner.py
"""

import cv2
import numpy as np

CAMERA_INDEX = 1   # change if needed

def nothing(x):
    pass   # trackbar callback requires a function — this one does nothing

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)

# Create windows
cv2.namedWindow("Original")
cv2.namedWindow("Mask")

# Create 6 trackbars — lower and upper for H, S, V
# Red lower range example starting point: H=0, S=80, V=50
# Blue starting point:                    H=100, S=80, V=50
cv2.createTrackbar("Lower H", "Mask", 0,   180, nothing)
cv2.createTrackbar("Lower S", "Mask", 80,  255, nothing)
cv2.createTrackbar("Lower V", "Mask", 50,  255, nothing)
cv2.createTrackbar("Upper H", "Mask", 10,  180, nothing)
cv2.createTrackbar("Upper S", "Mask", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Mask", 255, 255, nothing)

print("Adjust sliders until the target color is solid white, everything else black.")
print("Press S to print the current values. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read current slider positions
    lh = cv2.getTrackbarPos("Lower H", "Mask")
    ls = cv2.getTrackbarPos("Lower S", "Mask")
    lv = cv2.getTrackbarPos("Lower V", "Mask")
    uh = cv2.getTrackbarPos("Upper H", "Mask")
    us = cv2.getTrackbarPos("Upper S", "Mask")
    uv = cv2.getTrackbarPos("Upper V", "Mask")

    # Build and apply mask
    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])
    mask  = cv2.inRange(hsv, lower, upper)

    # Show both windows side by side
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Print values ready to paste into camera.py
        print(f"\n--- Current HSV values ---")
        print(f"lower = ({lh}, {ls}, {lv})")
        print(f"upper = ({uh}, {us}, {uv})")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()