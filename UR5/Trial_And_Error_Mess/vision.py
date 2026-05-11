"""Vision module for EV battery disassembly robot.

Provides camera acquisition, homography calibration, HSV masking,
and blob detection for identifying and localising battery modules.

Pipeline:
    1. Camera initialisation
    2. Live feed for orientation check
    3. Homography setup via 4 calibration target clicks
    4. Frame acquisition and cropping
    5. HSV conversion and binary masking
    6. Morphological cleanup
    7. Blob detection
    8. Centroid transformation to world coordinates
"""

import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

DEBUG_ENABLED = True

DEBUG_CONFIG = {
    'log_initialize': True,   # Logs resolution and focus on camera init.
    'show_frame': False,       # Shows raw frame on every get_frame call.
    'show_frame_shape': False, # Prints frame shape on every get_frame call.
    'show_mask': True,        # Shows binary mask during processing.
    'show_blobs': True,       # Shows blob detection result during processing.
    'live_feed': False,         # Shows live feed with crosshair on startup.
    'show_hsv': True,           # Shows HSV conversion.
    'show_mask_red': True,      # Shows red mask before combining.
    'show_mask_blue': True,     # Shows blue mask before combining.
    'show_mask_combined': True, # Shows combined mask before morphology.
    'show_mask_clean': True,    # Shows mask after morphology cleanup.
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_CONFIG = {
    'index': 1,               # Camera device index.
    'buffer_size': 1,         # Frame buffer size. Prevents stale frames.
    'focus_value': 0,         # Focus lock value. 0 = infinity focus.
    'frame_width': 1920,      # Capture resolution width in pixels.
    'frame_height': 1080,     # Capture resolution height in pixels.
}

HSV_CONFIG = {
    'red_lower_1': np.array([0, 100, 100]),    # Red hue lower bound near 0 degrees.
    'red_upper_1': np.array([10, 255, 255]),   # Red hue upper bound near 0 degrees.
    'red_lower_2': np.array([170, 100, 100]),  # Red hue lower bound near 360 degrees.
    'red_upper_2': np.array([180, 255, 255]),  # Red hue upper bound near 360 degrees.
    'blue_lower': np.array([100, 100, 50]),    # Blue hue lower bound.
    'blue_upper': np.array([130, 255, 255]),   # Blue hue upper bound.
}

BLOB_CONFIG = {
    'min_area': 500,    # Minimum blob area in pixels.
    'max_area': 50000,  # Maximum blob area in pixels.
}

MORPHOLOGY_CONFIG = {
    'erode_kernel_size': 5,   # Erosion kernel size. Removes noise.
    'dilate_kernel_size': 8,  # Dilation kernel size. Recovers blob size after erosion.
}

# World XY coordinates of the 4 calibration targets in mm.
# Targets must be placed at module surface height.
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
# Update if targets are moved.
CALIBRATION_TARGETS_WORLD = np.array([
    [480, 80],    # Top-Left target world position.
    [480, 320],    # Top-Right target world position.
    [640, 320],    # Bottom-Right target world position.
    [640, 80],    # Bottom-Left target world position.
], dtype=np.float32)

HOMOGRAPHY_SAVE_PATH = 'homography.npy'
CORNER_LABELS = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']


# ---------------------------------------------------------------------------
# Camera class
# ---------------------------------------------------------------------------

class Camera:
    """Represents a camera device or static image source for frame acquisition.

    Attributes:
        _cap: The cv2.VideoCapture instance. None if using static image.
        _test_image: Static image array. None if using live camera.
    """

    def __init__(self) -> None:
        """Initializes empty Camera instance. Use from_device() or from_image()."""
        self._cap = None
        self._test_image = None

    @classmethod
    def from_device(cls, camera_index: int, buffer_size: int) -> 'Camera':
        """Creates a Camera instance from a physical camera device.

        Args:
            camera_index: Index of the camera device to open.
            buffer_size: Number of frames to keep in the buffer.

        Returns:
            A Camera instance connected to the device.

        Raises:
            RuntimeError: If the camera could not be opened.
        """
        camera = cls()
        camera._cap = cv2.VideoCapture(camera_index)
        if not camera._cap.isOpened():
            raise RuntimeError('Could not open camera.')
        camera._cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)  # Prevents stale frames.
        camera._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Locks focus for stable focal length.
        camera._cap.set(cv2.CAP_PROP_FOCUS, CAMERA_CONFIG['focus_value'])
        camera._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['frame_width'])
        camera._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['frame_height'])
        for _ in range(5):  # Warmup frames to allow camera to initialise.
            camera._cap.grab()
        if DEBUG_ENABLED and DEBUG_CONFIG['log_initialize']:
            current_width = int(camera._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_height = int(camera._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f'Camera opened on index {camera_index} with buffer_size {buffer_size}.')
            print(f'Resolution: {current_width}x{current_height}.')
            print(f'Autofocus disabled, focus locked at {CAMERA_CONFIG["focus_value"]}.')
        return camera

    @classmethod
    def from_image(cls, image_path: str) -> 'Camera':
        """Creates a Camera instance from a static test image for offline use.

        Args:
            image_path: Path to the image file.

        Returns:
            A Camera instance loaded with the static image.

        Raises:
            ValueError: If the image could not be loaded.
        """
        camera = cls()
        camera._test_image = cv2.imread(image_path)
        if camera._test_image is None:
            raise ValueError(f'Could not load image from path: {image_path}.')
        if DEBUG_ENABLED and DEBUG_CONFIG['log_initialize']:
            print(f'Loaded test image from {image_path}.')
            print(f'Image shape: {camera._test_image.shape}.')
        return camera

    def get_frame(self) -> np.ndarray:
        """Retrieves a fresh frame from the camera or returns the test image.

        Returns:
            A BGR image as a numpy array of shape (height, width, 3).

        Raises:
            RuntimeError: If neither camera nor test image is available.
            RuntimeError: If the frame could not be retrieved from camera.
        """
        if self._test_image is not None:
            return self._test_image.copy()
        if self._cap is None:
            raise RuntimeError('Camera not initialised. Use from_device() or from_image().')
        self._cap.grab()
        ret, frame = self._cap.retrieve()
        if not ret:
            raise RuntimeError('Could not retrieve frame from camera.')
        if DEBUG_ENABLED and DEBUG_CONFIG['show_frame']:
            cv2.imshow('Debug: Raw Frame', frame)
            cv2.waitKey(1)
        if DEBUG_ENABLED and DEBUG_CONFIG['show_frame_shape']:
            print(f'Frame shape: {frame.shape}.')
        return frame

    def show_live_feed(self) -> None:
        """Displays a live camera feed with a centre crosshair until any key is pressed.

        Raises:
            RuntimeError: If the camera has not been initialised.
        """
        if self._cap is None and self._test_image is None:
            raise RuntimeError('Camera not initialised. Use from_device() or from_image().')
        print('Live feed active. Press any key to continue.')
        while cv2.waitKey(1) == -1:
            frame = self.get_frame()
            height, width = frame.shape[:2]
            cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
            cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 0), 1)
            cv2.imshow('Live Feed', frame)
        cv2.destroyAllWindows()

    def tune_focus(self) -> None:
        """Opens a window with a focus slider to find the correct focus lock value.

        Raises:
            RuntimeError: If the camera has not been initialised.
        """
        if self._cap is None:
            raise RuntimeError('Camera not initialised. Use from_device().')
        cv2.namedWindow('Focus Tuning')
        cv2.createTrackbar(
            'Focus',
            'Focus Tuning',
            CAMERA_CONFIG['focus_value'],
            255,
            lambda x: None,
        )
        while cv2.waitKey(1) != ord('q'):
            focus_value = cv2.getTrackbarPos('Focus', 'Focus Tuning')
            self._cap.set(cv2.CAP_PROP_FOCUS, focus_value)
            frame = self.get_frame()
            cv2.imshow('Focus Tuning', frame)
        cv2.destroyAllWindows()
        print(f'Focus tuning complete. Set CAMERA_CONFIG focus_value to: {focus_value}.')

    def release(self) -> None:
        """Releases the camera resource cleanly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# Homography class
# ---------------------------------------------------------------------------

class Homography:
    """Manages perspective transform between pixel and world coordinates.

    Uses 4 physical calibration targets at known world positions to compute
    a homography matrix H that maps pixel coordinates to world XY in mm.

    Attributes:
        _camera: Camera instance used for frame acquisition.
        _matrix: The 3x3 homography matrix H. None until setup is run.
    """

    def __init__(self, camera: Camera) -> None:
        """Initialises Homography with a Camera instance.

        Loads a saved homography matrix from file if one exists.

        Args:
            camera: Camera instance for frame acquisition.
        """
        self._camera = camera
        self._matrix = None
        if os.path.exists(HOMOGRAPHY_SAVE_PATH):
            self._matrix = np.load(HOMOGRAPHY_SAVE_PATH)
            print('Homography matrix loaded from file.')

    def setup(self) -> None:
        """Displays a frame and lets the user click 4 calibration targets.

        Offers to load a saved matrix if one exists. Saves the computed
        matrix to file after calibration.

        Raises:
            RuntimeError: If homography setup is cancelled by the user.
        """
        if self._matrix is not None:
            frame = self._camera.get_frame()
            cv2.putText(
                frame,
                'Saved homography found. Press L to load or C to recalibrate.',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow('Homography Setup', frame)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('l'):
                    cv2.destroyAllWindows()
                    print('Using saved homography matrix.')
                    return
                elif key == ord('c'):
                    cv2.destroyAllWindows()
                    print('Recalibrating homography.')
                    break

        base_frame = self._camera.get_frame()
        clicks = []
        mouse_pos = [None]

        def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
            if event == cv2.EVENT_MOUSEMOVE:
                mouse_pos[0] = (x, y)
            elif event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append((x, y))
                print(f'Click {len(clicks)}: {CORNER_LABELS[len(clicks) - 1]} at ({x}, {y}).')

        cv2.namedWindow('Homography Setup')
        cv2.setMouseCallback('Homography Setup', on_mouse)

        while True:
            display = self._draw_state(base_frame, clicks, mouse_pos[0])
            cv2.imshow('Homography Setup', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('u') and clicks:
                print(f'Undo: removed {clicks.pop()}.')
            elif key == ord('q'):
                cv2.destroyAllWindows()
                raise RuntimeError('Homography setup cancelled by user.')
            elif key != 255 and len(clicks) == 4:
                break

        cv2.destroyAllWindows()

        pixel_points = np.array(clicks, dtype=np.float32)
        self._matrix = cv2.getPerspectiveTransform(
            pixel_points,
            CALIBRATION_TARGETS_WORLD,
        )
        np.save(HOMOGRAPHY_SAVE_PATH, self._matrix)
        print('Homography saved to file.')
        print(f'Homography matrix:\n{self._matrix}')

    def transform_point(self, pixel_point: tuple[int, int]) -> tuple[float, float]:
        """Converts a single pixel coordinate to world XY in mm.

        Args:
            pixel_point: The (x, y) pixel coordinate to transform.

        Returns:
            A tuple of (world_x, world_y) in mm.

        Raises:
            RuntimeError: If homography has not been set up.
        """
        if self._matrix is None:
            raise RuntimeError('Homography not set up. Call setup() first.')
        point = np.array([[[float(pixel_point[0]), float(pixel_point[1])]]])
        result = cv2.perspectiveTransform(point, self._matrix)
        return float(result[0][0][0]), float(result[0][0][1])

    def transform_centroids(
        self,
        centroids: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Converts a list of pixel centroids to world XY coordinates in mm.

        Args:
            centroids: List of (x, y) pixel centroid coordinates.

        Returns:
            List of (world_x, world_y) tuples in mm.

        Raises:
            RuntimeError: If homography has not been set up.
        """
        if self._matrix is None:
            raise RuntimeError('Homography not set up. Call setup() first.')
        return [self.transform_point(c) for c in centroids]

    def verify(self) -> None:
        """Displays a live frame where clicking prints the world coordinate.

        Raises:
            RuntimeError: If homography has not been set up.
        """
        if self._matrix is None:
            raise RuntimeError('Homography not set up. Call setup() first.')
        print('Verification mode. Click anywhere to get world coordinates. Press Q to exit.')
        last_result = [None]

        def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                world = self.transform_point((x, y))
                last_result[0] = (x, y, world[0], world[1])
                print(f'Pixel ({x}, {y}) -> World ({world[0]:.1f}, {world[1]:.1f}) mm.')

        cv2.namedWindow('Homography Verification')
        cv2.setMouseCallback('Homography Verification', on_mouse)
        while cv2.waitKey(1) != ord('q'):
            frame = self._camera.get_frame()
            if last_result[0] is not None:
                px, py, wx, wy = last_result[0]
                cv2.putText(
                    frame,
                    f'Pixel ({px}, {py}) -> World ({wx:.1f}, {wy:.1f}) mm.',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            cv2.imshow('Homography Verification', frame)
        cv2.destroyAllWindows()

    def _draw_state(
        self,
        base_frame: np.ndarray,
        clicks: list[tuple[int, int]],
        mouse_pos: tuple[int, int] | None,
    ) -> np.ndarray:
        """Draws click state and rubber band line onto a copy of the frame.

        Args:
            base_frame: Clean unmodified frame to draw on.
            clicks: List of clicked pixel coordinates.
            mouse_pos: Current mouse position for rubber band line.

        Returns:
            A copy of the frame with all visual feedback drawn on it.
        """
        display = base_frame.copy()

        for i, point in enumerate(clicks):
            cv2.circle(display, point, 6, (0, 255, 0), -1)
            cv2.putText(
                display,
                CORNER_LABELS[i],
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            if i > 0:
                self._draw_dotted_line(display, clicks[i - 1], point, (0, 255, 0))

        if len(clicks) == 4:
            self._draw_dotted_line(display, clicks[3], clicks[0], (0, 165, 255))

        if clicks and mouse_pos is not None and len(clicks) < 4:
            self._draw_dotted_line(display, clicks[-1], mouse_pos, (0, 200, 255))

        label = (
            f'Click {CORNER_LABELS[len(clicks)]} | U: undo | Q: quit'
            if len(clicks) < 4
            else 'Press any key to confirm | U: undo | Q: quit'
        )
        cv2.putText(
            display,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return display

    @staticmethod
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
        if steps == 0:
            return
        for i in range(steps):
            t = i / steps
            x = int(pt1[0] + t * (pt2[0] - pt1[0]))
            y = int(pt1[1] + t * (pt2[1] - pt1[1]))
            cv2.circle(frame, (x, y), radius, color, -1)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class Detector:
    """Detects battery modules in a camera frame using HSV masking and blob detection.

    Attributes:
        _camera: Camera instance for frame acquisition.
        _homography: Homography instance for coordinate transformation.
    """

    def __init__(self, camera: Camera, homography: Homography) -> None:
        """Initialises Detector with Camera and Homography instances.

        Args:
            camera: Camera instance for frame acquisition.
            homography: Homography instance for coordinate transformation.
        """
        self._camera = camera
        self._homography = homography

    def detect(self) -> list[tuple[float, float]]:
        """Runs the full detection pipeline and returns world coordinates.

        Pipeline: get frame -> HSV -> mask -> morphology -> blobs -> world coords.

        Returns:
            List of (world_x, world_y) tuples in mm for each detected module.
        """
        frame = self._camera.get_frame()
        mask = self._build_mask(frame)
        mask = self._clean_mask(mask)
        centroids = self._get_centroids(mask)
        world_coords = self._homography.transform_centroids(centroids)
        if DEBUG_ENABLED and DEBUG_CONFIG['show_blobs']:
            self._draw_blobs(frame, centroids, world_coords)
        return world_coords

    def _build_mask(self, frame: np.ndarray) -> np.ndarray:
        """Converts frame to HSV and builds a combined red and blue binary mask.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Binary mask as numpy array.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask_1 = cv2.inRange(hsv, HSV_CONFIG['red_lower_1'], HSV_CONFIG['red_upper_1'])
        red_mask_2 = cv2.inRange(hsv, HSV_CONFIG['red_lower_2'], HSV_CONFIG['red_upper_2'])
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        blue_mask = cv2.inRange(hsv, HSV_CONFIG['blue_lower'], HSV_CONFIG['blue_upper'])
        mask = cv2.bitwise_or(red_mask, blue_mask)
        if DEBUG_ENABLED and DEBUG_CONFIG['show_mask']:
            cv2.imshow('Debug: Binary Mask', mask)
            cv2.waitKey(1)
        return mask

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Applies morphological erosion and dilation to clean the binary mask.

        Args:
            mask: Binary mask as numpy array.

        Returns:
            Cleaned binary mask as numpy array.
        """
        erode_kernel = np.ones(
            (MORPHOLOGY_CONFIG['erode_kernel_size'],
             MORPHOLOGY_CONFIG['erode_kernel_size']),
            np.uint8,
        )
        dilate_kernel = np.ones(
            (MORPHOLOGY_CONFIG['dilate_kernel_size'],
             MORPHOLOGY_CONFIG['dilate_kernel_size']),
            np.uint8,
        )
        mask = cv2.erode(mask, erode_kernel)    # Removes noise.
        mask = cv2.dilate(mask, dilate_kernel)  # Recovers blob size after erosion.
        return mask

    def _get_centroids(self, mask: np.ndarray) -> list[tuple[float, float]]:
        """Detects blobs in the mask and returns their pixel centroids.

        Args:
            mask: Binary mask as numpy array.

        Returns:
            List of (x, y) pixel centroid coordinates.
        """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = BLOB_CONFIG['min_area']
        params.maxArea = BLOB_CONFIG['max_area']
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(cv2.bitwise_not(mask))  # Detector needs white blobs on black.
        return [(kp.pt[0], kp.pt[1]) for kp in keypoints]

    def _draw_blobs(
        self,
        frame: np.ndarray,
        centroids: list[tuple[float, float]],
        world_coords: list[tuple[float, float]],
    ) -> None:
        """Draws detected blobs and their world coordinates on the frame.

        Args:
            frame: BGR image to draw on.
            centroids: List of pixel centroid coordinates.
            world_coords: List of world XY coordinates in mm.
        """
        display = frame.copy()
        for (px, py), (wx, wy) in zip(centroids, world_coords):
            cx, cy = int(px), int(py)
            cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(
                display,
                f'({wx:.1f}, {wy:.1f})',
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        cv2.imshow('Debug: Detected Modules', display)
        cv2.waitKey(1)


# Test Helper Function
def _wait_for_key(message: str = 'Press any key to continue...') -> None:
    """Displays a message on a status window and waits for a keypress.

    Args:
        message: Message to print before waiting.
    """
    print(f'\n{message}')
    status = np.zeros((60, 500, 3), dtype=np.uint8)
    cv2.putText(
        status,
        message,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.imshow('Test Status', status)
    while cv2.waitKey(100) == -1:
        pass

# ---------------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('--- Vision Module Test ---')

    # Test 1: Camera from device
    print('\n[1] Initialising camera from device...')
    camera = Camera.from_device(
        CAMERA_CONFIG['index'],
        CAMERA_CONFIG['buffer_size'],
    )
    _wait_for_key()

    # Test 2: Live feed
    if DEBUG_ENABLED and DEBUG_CONFIG['live_feed']:
        print('\n[2] Starting live feed. Press any key to continue.')
        camera.show_live_feed()
        _wait_for_key()

    # Test 3: Single frame
    print('\n[3] Grabbing test frame...')
    frame = camera.get_frame()
    print(f'Frame shape: {frame.shape}.')
    cv2.imshow('Test: Raw Frame', frame)
    cv2.waitKey(1)
    _wait_for_key()

    # Test 4: Homography setup
    print('\n[4] Setting up homography...')
    homography = Homography(camera)
    homography.setup()
    _wait_for_key()

    # Test 5: Transform a single point
    print('\n[5] Testing single point transform...')
    test_pixel = (960, 540)  # Image centre.
    world = homography.transform_point(test_pixel)
    print(f'Pixel {test_pixel} -> World ({world[0]:.1f}, {world[1]:.1f}) mm.')
    _wait_for_key()

    # Test 6: Homography verification
    print('\n[6] Homography verification. Click to test. Press Q to exit.')
    homography.verify()
    _wait_for_key()

    # Test 7: Detector pipeline step by step
    print('\n[7] Running detector pipeline step by step...')
    detector = Detector(camera, homography)
    frame = camera.get_frame()

    print('  [7a] Converting to HSV...')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('Step 7a: HSV', hsv)
    cv2.waitKey(1)
    _wait_for_key()

    print('  [7b] Building mask...')
    mask = detector._build_mask(frame)
    cv2.imshow('Step 7b: Combined Mask', mask)
    cv2.waitKey(1)
    _wait_for_key()

    print('  [7c] Cleaning mask with morphology...')
    mask_clean = detector._clean_mask(mask)
    cv2.imshow('Step 7c: Clean Mask', mask_clean)
    cv2.waitKey(1)
    _wait_for_key()

    print('  [7d] Detecting blob centroids...')
    centroids = detector._get_centroids(mask_clean)
    print(f'  Detected {len(centroids)} blob(s).')
    for i, (px, py) in enumerate(centroids):
        print(f'    Blob {i + 1}: Pixel ({px:.1f}, {py:.1f}).')
    _wait_for_key()

    print('  [7e] Transforming centroids to world coordinates...')
    world_coords = homography.transform_centroids(centroids)
    for i, (wx, wy) in enumerate(world_coords):
        print(f'    Module {i + 1}: World ({wx:.1f}, {wy:.1f}) mm.')
    detector._draw_blobs(frame, centroids, world_coords)
    cv2.waitKey(1)
    _wait_for_key()

    # Test 8: Camera from static image
    print('\n[8] Testing offline mode with static image...')
    cv2.imwrite('test_frame.jpg', fra me)
    offline_camera = Camera.from_image('test_frame.jpg')
    offline_frame = offline_camera.get_frame()
    print(f'Offline frame shape: {offline_frame.shape}.')
    offline_homography = Homography(offline_camera)
    offline_detector = Detector(offline_camera, offline_homography)
    offline_coords = offline_detector.detect()
    print(f'Offline detected {len(offline_coords)} module(s).')
    for i, (wx, wy) in enumerate(offline_coords):
        print(f'  Module {i + 1}: World ({wx:.1f}, {wy:.1f}) mm.')
    _wait_for_key()

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    print('\n--- Test complete ---')