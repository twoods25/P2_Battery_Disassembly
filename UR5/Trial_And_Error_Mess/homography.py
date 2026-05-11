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
import os                                   # Interact with OS


#Constants
DEBUG_ENABLED = True
DEBUG_CONFIG = {
    'log_initialize': True,                 # Logs camera index, buffer, resolution, focus on init
    'log_frame_grab': False,
    'live_feed': True,
    'show_frame': False,
    'show_frame_shape': False,
}
CALIBRATION_CONFIG = {
    'focus_value':  0,
    'frame_width':  1920,
    'frame_height': 1080,
    'focal_length': 1344,
    'crop_x': 705,          # Top-left x of battery box region
    'crop_y': 270,          # Top-left y of battery box region
    'crop_width': 800,      # Width of crop region
    'crop_height': 600,     # Height of crop region
    'display_scale': 1.0,
    'module_surface_z': 31,  # Height of module surface above Z=0 in mm.
}
CAMERA_POSITION_XYZ = np.array([526, 160, 982])
CORNER_LABELS = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
HOMOGRAPHY_SAVE_PATH = 'homography.npy'


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
    cap = cv2.VideoCapture(camera_index)
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
       raise RuntimeError('Camera not initialized, call initialize() first.')
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

def crop_to_battery(frame: np.ndarray) -> np.ndarray:
    """Crops frame to the fixed battery box region.

    Args:
        frame: Full resolution camera frame.

    Returns:
        Cropped frame containing only the battery box region.
    """
    x = CALIBRATION_CONFIG['crop_x']
    y = CALIBRATION_CONFIG['crop_y']
    w = CALIBRATION_CONFIG['crop_width']
    h = CALIBRATION_CONFIG['crop_height']
    return frame[y:y + h, x:x + w]


#Homography Setup
def _draw_dotted_line(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    gap: int = 10,
    radius: int = 2,
) -> None:
    """Draws a dotted line between two points on a frame.

    Args:
        frame: The image to draw on.
        pt1: Start point as (x, y).
        pt2: End point as (x, y).
        color: BGR color tuple.
        gap: Pixel spacing between dots.
        radius: Radius of each dot in pixels.
    """
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    steps = int(dist / gap)
    for i in range(steps):
        t = i / steps
        x = int(pt1[0] + t * (pt2[0] - pt1[0]))
        y = int(pt1[1] + t * (pt2[1] - pt1[1]))
        cv2.circle(frame, (x, y), radius, color, -1)


def _draw_homography_state(
    base_frame: np.ndarray,
    clicks: list[tuple[int, int]],
    mouse_pos: tuple[int, int] | None,
    scale: float,
) -> np.ndarray:
    """Draws current click state and rubber band line onto a copy of the frame.

    Args:
        base_frame: Clean unmodified cropped and scaled frame to draw on.
        clicks: List of clicked points in original frame coordinates.
        mouse_pos: Current mouse position in display coordinates.
        scale: Display scale factor used to convert coordinates for drawing.

    Returns:
        A copy of the frame with all visual feedback drawn on it.
    """
    display = base_frame.copy()
    color_dot = (0, 255, 0)
    color_line = (0, 255, 0)
    color_rubber = (0, 200, 255)
    color_close = (0, 165, 255)

    def to_display(pt: tuple[int, int]) -> tuple[int, int]:
        """Converts original frame coordinates to display coordinates."""
        dx = int((pt[0] - CALIBRATION_CONFIG['crop_x']) * scale)
        dy = int((pt[1] - CALIBRATION_CONFIG['crop_y']) * scale)
        return dx, dy

    for i, point in enumerate(clicks):
        dp = to_display(point)
        cv2.circle(display, dp, 6, color_dot, -1)
        cv2.putText(
            display,
            CORNER_LABELS[i],
            (dp[0] + 10, dp[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_dot,
            2,
        )
        if i > 0:
            _draw_dotted_line(display, to_display(clicks[i - 1]), dp, color_line)

    if len(clicks) == 4:
        _draw_dotted_line(display, to_display(clicks[3]), to_display(clicks[0]), color_close)

    if clicks and mouse_pos is not None and len(clicks) < 4:
        _draw_dotted_line(display, to_display(clicks[-1]), mouse_pos, color_rubber)

    if len(clicks) < 4:
        cv2.putText(
            display,
            f'Click {CORNER_LABELS[len(clicks)]} | U: undo | Q: quit',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    else:
        cv2.putText(
            display,
            'Press any key to confirm | U: undo | Q: quit',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return display


def _clicks_to_world(
    clicks: list[tuple[int, int]],
) -> np.ndarray:
    """Converts 4 pixel clicks to world XY coordinates.

    Args:
        clicks: List of 4 (x, y) pixel coordinates.

    Returns:
        A numpy array of shape (4, 2) with world XY coordinates in mm.
    """
    focal_length = CALIBRATION_CONFIG['focal_length']
    camera_x, camera_y, camera_z = CAMERA_POSITION_XYZ
    module_z = CALIBRATION_CONFIG['module_surface_z']  # 27.08mm
    effective_z = camera_z - module_z                   # Actual distance to module surface.
    image_cx = CALIBRATION_CONFIG['frame_width'] / 2
    image_cy = CALIBRATION_CONFIG['frame_height'] / 2
    world_points = []
    for px, py in clicks:
        world_x = camera_x + (py - image_cy) * (effective_z / focal_length)
        world_y = camera_y + (px - image_cx) * (effective_z / focal_length)
        world_points.append([world_x, world_y])
    return np.array(world_points, dtype=np.float32)


def setup_homography() -> np.ndarray:
    """Captures a frame and lets the user click 4 corners to compute homography.

    Checks for a saved homography file first and offers to load it.

    Returns:
        A 3x3 perspective transform matrix H.

    Raises:
        RuntimeError: If the camera has not been initialized.
        RuntimeError: If homography setup was cancelled by the user.
    """
    if cap is None:
        raise RuntimeError('Camera not initialized, call initialize() first.')

    if os.path.exists(HOMOGRAPHY_SAVE_PATH):
        frame = get_frame()
        cropped = crop_to_battery(frame)
        scale = CALIBRATION_CONFIG['display_scale']
        display = cv2.resize(cropped, (0, 0), fx=scale, fy=scale)
        cv2.putText(
            display,
            'Saved homography found. Press L to load or C to recalibrate.',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow('Homography Setup', display)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('l'):
                cv2.destroyAllWindows()
                print('Homography loaded from file.')
                return np.load(HOMOGRAPHY_SAVE_PATH)
            elif key == ord('c'):
                cv2.destroyAllWindows()
                print('Recalibrating homography.')
                break

    base_frame = get_frame()
    cropped = crop_to_battery(base_frame)
    scale = CALIBRATION_CONFIG['display_scale']
    base_display = cv2.resize(cropped, (0, 0), fx=scale, fy=scale)

    clicks = []
    mouse_pos = [None]

    def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            orig_x = int(x / scale + CALIBRATION_CONFIG['crop_x'])
            orig_y = int(y / scale + CALIBRATION_CONFIG['crop_y'])
            clicks.append((orig_x, orig_y))
            print(f'Click {len(clicks)}: {CORNER_LABELS[len(clicks) - 1]} at ({orig_x}, {orig_y}).')

    cv2.namedWindow('Homography Setup')
    cv2.setMouseCallback('Homography Setup', on_mouse)

    while True:
        display = _draw_homography_state(base_display, clicks, mouse_pos[0], scale)
        cv2.imshow('Homography Setup', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('u') and clicks:
            removed = clicks.pop()
            print(f'Undo: removed {removed}.')
        elif key == ord('q'):
            cv2.destroyAllWindows()
            raise RuntimeError('Homography setup cancelled by user.')
        elif key != 255 and len(clicks) == 4:
            break

    cv2.destroyAllWindows()

    world_points = _clicks_to_world(clicks)
    pixel_points = np.array(clicks, dtype=np.float32)
    homography_matrix = cv2.getPerspectiveTransform(pixel_points, world_points)

    np.save(HOMOGRAPHY_SAVE_PATH, homography_matrix)
    print('Homography saved to file.')
    print(f'Homography matrix:\n{homography_matrix}')

    return homography_matrix

def apply_homography(
    homography_matrix: np.ndarray,
    pixel_point: tuple[int, int],
) -> tuple[float, float]:
    """Converts a pixel coordinate to world XY coordinates using homography.

    Args:
        homography_matrix: The 3x3 perspective transform matrix H.
        pixel_point: The (x, y) pixel coordinate to transform.

    Returns:
        A tuple of (world_x, world_y) in mm.
    """
    point = np.array([[[float(pixel_point[0]), float(pixel_point[1])]]])
    world_point = cv2.perspectiveTransform(point, homography_matrix)
    return float(world_point[0][0][0]), float(world_point[0][0][1])


# Calibration Functions
def show_live_feed() -> None:
    """Displays a live camera feed with a center crosshair until any key is pressed.

    Raises:
        RuntimeError: If the camera has not been initialized.
    """
    if cap is None:
        raise RuntimeError('Camera not initialized, call initialize() first.')
    print('Live feed active. Press any key to continue.')
    while cv2.waitKey(1) == -1:
        frame = get_frame()
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (0, 255, 0), 1)
        cv2.imshow('Live Feed', frame)
    cv2.destroyAllWindows()

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

def verify_homography(homography_matrix: np.ndarray) -> None:
    """Displays a live frame where clicking prints the world coordinate.

    Args:
        homography_matrix: The 3x3 perspective transform matrix H.

    Raises:
        RuntimeError: If the camera has not been initialized.
    """
    if cap is None:
        raise RuntimeError('Camera not initialized, call initialize() first.')

    print('Verification mode. Click anywhere to get world coordinates. Press Q to exit.')
    last_result = [None]

    def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            world = apply_homography(homography_matrix, (x, y))
            last_result[0] = (x, y, world[0], world[1])
            print(f'Pixel ({x}, {y}) -> World ({world[0]:.1f}, {world[1]:.1f}) mm.')

    cv2.namedWindow('Homography Verification')
    cv2.setMouseCallback('Homography Verification', on_mouse)

    while cv2.waitKey(1) != ord('q'):
        frame = get_frame()
        if last_result[0] is not None:
            px, py, wx, wy = last_result[0]
            cv2.putText(frame, f'Pixel ({px}, {py}) -> World ({wx:.1f}, {wy:.1f}) mm.',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Homography Verification', frame)
    cv2.destroyAllWindows()


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
    initialize(1, 1)
    if DEBUG_ENABLED and DEBUG_CONFIG['live_feed']:
        show_live_feed()
    homography_matrix = setup_homography()
    print(apply_homography(homography_matrix, (842, 501)))
    verify_homography(homography_matrix)
    #tune_focus()
    #get_resolution()
    get_frame()
    cv2.waitKey(0)
    cv2.destroyAllWindows()