import cv2
from PIL import Image
from util import get_limits

red = [0, 0, 255] #Yellow in BGR colour space.
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert our image from BGR colour space to HSV colour space.

    lowerLimit, upperLimit = get_limits(color=red)
    
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit) #Mask from all pixels that belong to the colour we want to detect.

    mask_ = Image.fromarray(mask) #Converting image from numpy array to pillow. Keeping information, but in different format.

    bbox = mask_.getbbox() #Function from Pillow. This function gets bonding box.

    if bbox is not None:                
        x1, y1, x2, y2 = bbox       #Get location

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5) #Draw rectangle with bbox. Drawing on frame, specify upper left, specify bottom right, then colour, then thickness

    print(bbox)

    cv2.imshow('frame', frame) #Changed Mask to drawing the frame.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()