"""
Module containing vision functions
"""
#
"""Import of Libraries"""
import cv2                                  #Computer Vision Library
import numpy as np                          #Numpy Library
#
"""Initialize variables"""
cap = None                                  #Declaring variable for video capture.
#
"""Vision Functions"""
def initialize():                           #Initialize and establish stream to Camera
    """
    Initializes the camera so its ready for use
    """
    global cap                              #Writes to variable outside function
    cap = cv2.VideoCapture(1)               #Open Stream to Robot Camera
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     #Keeps only one image buffered -> Fresh image every time
    if not cap.isOpened():                  #EH: Checks Camera Connection
        raise Exception("Could not open camera")
#
def get_frame():
    """
    Retrieves a frame from the camera and decodes it into BGR
    """
    if cap is None:                         #EH: Ensures that initialize function has been called first
       raise Exception("Camera not initialized, call initialize() first")
    cap.grab()                              #Claims a frame from the buffer
    ret, frame = cap.retrieve()             #Retrieves frame (bool, numpy array)
    if not ret:                             #EH: ret = 0 -> Did not return frame.
        raise Exception("Could not retrieve frame")
    return frame                            #Returns numpy array "frame"
#
def red_mask(hsv_image):
    """
    Takes an HSV image as input and returns a red binary mask
    """
    red1_lower = np.array([1, 100, 184])                 #Red lower at 0 degrees HSV
    red1_upper = np.array([10, 255, 255])               #Red upper at 0 degrees HSV
    #red2_lower = np.array([170, 120, 70])               #Red lower at 360 degrees HSV
    #red2_upper = np.array([180, 255, 255])              #Red upper at 360 degrees HSV
    submask1 = cv2.inRange(hsv_image, red1_lower, red1_upper)       #Submask near 0 degrees HSV
    #submask2 = cv2.inRange(hsv_image, red2_lower, red2_upper)       #Submask near 360 degrees HSV
    #red_mask = cv2.bitwise_or(submask1, submask2)                   #Combining submask to one mask
    return submask1 #red_mask                                                 #Returns red mask
#
def blob_detector(red_mask_clean):
    """
    Takes the cleaned up mask, and outputs detected modules.
    The mask is inverted, and a black blob cv2 detector is applied.
    Unnecessary parameters from cv2 blob is turned off.
    """
    blob_parameters = cv2.SimpleBlobDetector_Params()       #Creates parameters from blueprint
    blob_parameters.filterByArea = True     # Turning on filter by area
    blob_parameters.minArea = 500           # minimum blob size in pixels
    blob_parameters.maxArea = 50000         # maximum blob size in pixels
    blob_parameters.filterByCircularity = False     # Turning off circularity filter
    blob_parameters.filterByConvexity = False       # Turning off convexity filter
    blob_parameters.filterByInertia = False         # Turning off inertia filter
    #
    detector = cv2.SimpleBlobDetector_create(blob_parameters)    # Create blob detector with parameters
    inverted_mask = cv2.bitwise_not(red_mask_clean)     # Converts black to white and white to black (inverses binary image)
    keypoints = detector.detect(inverted_mask)
    #
    return keypoints            # Blob information returned
#
def process(frame):
    """
    Takes a BGR image as input and returns color detection array.
    Pipeline: BGR to HSV -> Binary Mask -> Morphology -> Blob Detection -> Mapping -> Return Position Array
    """
    #hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    #Convert BGR to HSV color space
    #red_mask = red_mask(hsv_image)  #Create red mask (NOTE: Could change to blue for easier detection)
    #red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)   #Morphological cleanup (remove noise)
    #blob_detect = blob_detector(red_mask_clean)    #Detect blobs on mask
    #Map blobs to positions
    return hsv_image        #Return result array

"""Test Functions"""
def frame_check(frame):
    """
    Test function to check an image during program runtime.
    Pauses at image, click any key to continue.
    """
    cv2.imshow("Status", frame)  #
    cv2.waitKey(0)  # wait until any key is pressed
#
def visualize_blobs(image, keypoints):
    """
    Visualize the BGR image with blob detection.
    """
    image_copy = image.copy()   # Copies the image for drawing on
    for kp in keypoints:        # Creates a list for drawing blobs
        print(kp.pt)            # (x, y) center position of blob
        print(kp.size)          # diameter of blob
        x, y = int(kp.pt[0]), int(kp.pt[1])     #Center points
        size = int(kp.size/2)                   #Diameter to radius
        cv2.rectangle(                          #Create rectangle
            image_copy,
            (x - size, y - size),               #Top left corner
            (x + size, y + size),               #Bottom right corner
            (0, 255, 0),                        #Green color marking
            2                                   #Outline thickness
        )
        cv2.imshow("Detected modules", image_copy)
        cv2.waitKey(0)
#
"""Test of vision program"""
ini = initialize()
bgr_image = get_frame()
frame_check(bgr_image)
hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
frame_check(hsv_image)
#
red1_lower = np.array([1, 100, 184])
red1_upper = np.array([10, 255, 255])
red2_lower = np.array([170, 120, 70])
red2_upper = np.array([180, 255, 255])
submask1 = cv2.inRange(hsv_image, red1_lower, red1_upper)
frame_check(submask1)
submask2 = cv2.inRange(hsv_image, red2_lower, red2_upper)
frame_check(submask2)
red_mask = cv2.bitwise_or(submask1, submask2)
frame_check(red_mask)
result = cv2.bitwise_and(bgr_image, bgr_image, mask=red_mask)
cv2.imshow("Result", result)
cv2.waitKey(0)
#
#kernel = np.ones((9, 9), np.uint8)
#red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
erode_kernel = np.ones((5, 5), np.uint8)
dilate_kernel = np.ones((8, 8), np.uint8)
red_mask_clean = cv2.erode(red_mask, erode_kernel)      # aggressive shrink
red_mask_clean = cv2.dilate(red_mask_clean, dilate_kernel)  # gentle recover
frame_check(red_mask_clean)
result = cv2.bitwise_and(bgr_image, bgr_image, mask=red_mask_clean)
frame_check(result)
keypoints = blob_detector(red_mask_clean)
visualize_blobs(bgr_image, keypoints)

"""End"""
cv2.destroyAllWindows()                     #Terminates all CV windows

