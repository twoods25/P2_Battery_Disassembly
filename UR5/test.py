"""
Module containing vision functions
"""
#
"""Import of Libraries"""
import cv2                                  #Computer Vision Library
#
"""Initialize variables"""
cap = None                                  #Declaring variable for video capture.
#
"""Vision Functions"""
def initialize():                           #Initialize and establish stream to Camera
    global cap                              #Writes to variable outside function
    cap = cv2.VideoCapture(1)               #Open Stream to Robot Camera
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     #Keeps only one image buffered -> Fresh image every time
    if not cap.isOpened():                  #EH: Checks Camera Connection
        raise Exception("Could not open camera")
#
def get_frame():
    if cap is None:                         #EH: Ensures that initialize function has been called first
       raise Exception("Camera not initialized, call initialize() first")
    cap.grab()                              #Claims a frame from the buffer
    ret, frame = cap.retrieve()             #Retrieves frame (bool, numpy array)
    if not ret:                             #EH: ret = 0 -> Did not return frame.
        raise Exception("Could not retrieve frame")
    return frame                            #Returns numpy array "frame"
#
"""Test Functions"""
def frame_check(frame):
    cv2.imshow("Status", frame)  #
    cv2.waitKey(0)  # wait until any key is pressed
#
#Test of vision program:
initialize()
frame = get_frame()
frame_check(frame)
#
#
"""End"""
cv2.destroyAllWindows()                     #Terminates all CV windows

