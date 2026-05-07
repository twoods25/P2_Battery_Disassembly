"""
Module for testing and setting up homography
1. cv2.VideoCapture → get frame
2. hardcode camera_xyz
3. get/estimate focal length
4. cv2.imshow + cv2.setMouseCallback → collect 4 clicks
5. convert clicks to world coords using camera xyz + focal length
6. cv2.getPerspectiveTransform → H
"""


# Import of Libraries
import cv2                                  # Computer Vision Library
import numpy as np                          # Numpy Library


#Constants
DEBUG_ENABLED = True

DEBUG_CONFIG = {
    'log_initialize': True,                 # Logs camera index, buffer, resolution, focus on init
    'log_frame_grab': False,
    'show_frame': True,
    'show_frame_shape': True,
}

CALIBRATION_CONFIG = {
    'focus_value': 0,
    'frame_width': 1280,
    'frame_height': 720,
}

CAMERA_POSITION_XYZ = np.array([640, 150, 445])


#Initialization
cap = None

# Camera Functions
def initialize(camera_index: int, buffer_size: int) -> None:
    """Initializes camera with fixed resolution, buffer size and focus.

    Args:
        camera_index: Index of the camera device to open.
        buffer_size: Number of frames to keep in the buffer.

    Raises:
        ValueError: If the camera could not be opened.
    """
    global cap                                          # Side-note: Needs to be change acc. styleguide
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError('Could not open camera')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)       # Prevents stale frames
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)                  #Locks focus for focal length.
    cap.set(cv2.CAP_PROP_FOCUS, CALIBRATION_CONFIG['focus_value'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIBRATION_CONFIG['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIBRATION_CONFIG['frame_height'])
    for _ in range(5):  # Warmup frames to allow DSHOW to initialize.
        cap.grab()
    if DEBUG_ENABLED and DEBUG_CONFIG['log_initialize']:
        current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Camera opened on index {camera_index} with buffer_size {buffer_size}.')
        print(f'Resolution: {current_width}x{current_height}.')
        print(f'Autofocus disabled, focus locked at {CALIBRATION_CONFIG["focus_value"]}.')

def get_frame() -> np.ndarray:
    """Retrieves a frame from the camera as BGR numpy array.

    Returns:
        A BGR image as a numpy array of shape (height, width, 3).

    Raises:
        RuntimeError: If the camera has not been initialized.
        RuntimeError: If the frame could not be retrieved.
    """
    if cap is None:
       raise RuntimeError('Camera not initialized, call initialize() first')
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        raise RuntimeError('Could not retrieve frame')
    if DEBUG_ENABLED and DEBUG_CONFIG['show_frame']:
        cv2.imshow('Debug: Raw Frame',frame)
        cv2.waitKey(1)
    if DEBUG_ENABLED and DEBUG_CONFIG['show_frame_shape']:
        print(f'Frame shape: {frame.shape}')
    return frame


# Calibration Functions
def get_focus_value() -> float:
    """Retrieves the current focus lock value from the camera.

    Returns:
        The current focus lock value.

    Raises:
        RuntimeError: If the camera has not been initialized.
        RuntimeError: If the focus value could not be retrieved.
    """
    if cap is None:
        raise RuntimeError('Camera not initialized, call initialize() first.')
    focus_value = cap.get(cv2.CAP_PROP_FOCUS)
    if focus_value < 0:
        raise RuntimeError('Could not retrieve focus value from camera.')
    print(f'Current focus value: {focus_value}.')
    return focus_value

def tune_focus() -> None:
    """Opens a window with a focus slider to find the correct focus value.

    Raises:
        RuntimeError: If the camera has not been initialized.
    """
    if cap is None:
        raise RuntimeError('Camera not initialized, call initialize() first.')

    cv2.namedWindow('Focus Tuning')
    cv2.createTrackbar(
        'Focus',
        'Focus Tuning',
        CALIBRATION_CONFIG['focus_value'],
        255,
        lambda x: None,
    )
    while cv2.waitKey(1) != ord('q'):
        focus_value = cv2.getTrackbarPos('Focus', 'Focus Tuning')
        cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        frame = get_frame()
        cv2.imshow('Focus Tuning', frame)
    cv2.destroyAllWindows()
    print(f'Focus tuning complete. Set CALIBRATION_CONFIG focus_value to: {focus_value}.')

def get_resolution() -> tuple[int, int]:
    """Retrieves the current camera resolution.

    Returns:
        A tuple of (width, height) in pixels.

    Raises:
        RuntimeError: If the camera has not been initialized.
    """
    if cap is None:
        raise RuntimeError('Camera not initialized, call initialize() first.')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera resolution: {width}x{height}.')
    return width, height


# Template
def function_name(param_one: int, param_two: str | None = None) -> int:
    """One-line summary ending with a period.

    Longer description if needed. Explain what the function does,
    not how it does it.

    Args:
        param_one: Description of param_one.
        param_two: Description of param_two. Defaults to None.

    Returns:
        Description of the return value.

    Raises:
        ValueError: If param_one is invalid.
    """


# TEMP FUNCTIONS



if __name__ == '__main__':
    initialize(1,1)
    #tune_focus()
    #get_resolution()
    get_frame()
    cv2.waitKey(0)
    cv2.destroyAllWindows()