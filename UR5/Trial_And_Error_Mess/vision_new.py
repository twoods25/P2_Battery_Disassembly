"""Vision module for EV battery disassembly robot.

Provides camera acquisition, homography calibration, HSV masking,
and blob detection for identifying and localising battery modules.

Pipeline:
    1. Camera initialisation
    2. Homography setup via 4 calibration target clicks on cropped frame
    3. Frame acquisition and cropping to battery work area
    4. HSV conversion and binary masking
    5. Morphological cleanup
    6. Blob detection
    7. Centroid transformation to world coordinates
    8. Colour classification per detected module
"""

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

DEBUG_ENABLED = True

DEBUG_CONFIG = {
    'log_initialize': True,    # Logs resolution and focus on camera init.
    'show_frame': False,       # Shows raw frame on every get_frame call.
    'show_frame_shape': False, # Prints frame shape on every get_frame call.
    'show_mask': False,        # Shows binary mask during processing.
    'show_blobs': False,       # Shows blob detection result during processing.
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_CONFIG = {
    'index': 1,             # Camera device index.
    'buffer_size': 1,       # Frame buffer size. Prevents stale frames.
    'focus_value': 0,       # Focus lock value. 0 = infinity focus.
    'frame_width': 1920,    # Capture resolution width in pixels.
    'frame_height': 1080,   # Capture resolution height in pixels.
    'crop_x': 705,          # Top-left x of battery work area in full frame.
    'crop_y': 270,          # Top-left y of battery work area in full frame.
    'crop_width': 800,      # Width of battery work area in pixels.
    'crop_height': 600,     # Height of battery work area in pixels.
}

HSV_CONFIG = {
    'red_lower_1': np.array([0, 100, 100]),    # Red hue lower bound near 0 degrees.
    'red_upper_1': np.array([10, 255, 255]),   # Red hue upper bound near 0 degrees.
    'red_lower_2': np.array([170, 100, 100]),  # Red hue lower bound near 360 degrees.
    'red_upper_2': np.array([180, 255, 255]),  # Red hue upper bound near 360 degrees.
    'blue_lower': np.array([100, 100, 50]),    # Blue hue lower bound.
    'blue_upper': np.array([130, 255, 255]),   # Blue hue upper bound.
}

MORPHOLOGY_CONFIG = {
    'erode_kernel_size': 5,  # Erosion kernel size. Removes noise.
    'dilate_kernel_size': 8, # Dilation kernel size. Recovers blob size after erosion.
}

BLOB_CONFIG = {
    'min_area': 500,   # Minimum blob area in pixels.
    'max_area': 50000, # Maximum blob area in pixels.
}

# World XY coordinates of the 4 calibration targets in mm.
# Targets must be placed at module surface height.
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
# Update if targets are moved.
CALIBRATION_TARGETS_WORLD = np.array([
    [480, 32],   # Top-Left target world position.
    [480, 320],  # Top-Right target world position.
    [640, 320],  # Bottom-Right target world position.
    [640, 32],   # Bottom-Left target world position.
], dtype=np.float32)

HOMOGRAPHY_SAVE_PATH = 'homography.npy'
CALIBRATION_PARAMS_PATH = 'calibration_params.json'
CORNER_LABELS = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
MAX_MODULES = 8  # Maximum number of modules in a battery pack.


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Holds the result of a single detection pass.

    Attributes:
        positions: List of (world_x, world_y) tuples in mm for each module.
        is_red: List of booleans. True if the module is red, False if blue.
    """

    positions: list
    is_red: list


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
        """Initialises empty Camera instance. Use from_device() or from_image()."""
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
        camera._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Locks focus for stable image.
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
        """Retrieves a fresh full frame from the camera or returns the test image.

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

    def get_cropped_frame(self) -> np.ndarray:
        """Retrieves a frame cropped to the battery work area.

        Returns:
            A BGR image cropped to the battery work area.

        Raises:
            RuntimeError: If the frame could not be retrieved.
        """
        frame = self.get_frame()
        x = CAMERA_CONFIG['crop_x']
        y = CAMERA_CONFIG['crop_y']
        w = CAMERA_CONFIG['crop_width']
        h = CAMERA_CONFIG['crop_height']
        return frame[y:y + h, x:x + w]

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
    a homography matrix H that maps cropped frame pixel coordinates to
    world XY in mm.

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
        """Displays cropped frame and lets the user click 4 calibration targets.

        Offers to load a saved matrix if one exists. Saves the computed
        matrix to file after calibration. All pixel coordinates are in
        the cropped frame coordinate system.

        Raises:
            RuntimeError: If homography setup is cancelled by the user.
        """
        if self._matrix is not None:
            frame = self._camera.get_cropped_frame()
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

        base_frame = self._camera.get_cropped_frame()
        clicks = []
        mouse_pos = [None]

        def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
            if event == cv2.EVENT_MOUSEMOVE:
                mouse_pos[0] = (x, y)
            elif event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append((x, y))
                print(
                    f'Click {len(clicks)}: {CORNER_LABELS[len(clicks) - 1]}'
                    f' at ({x}, {y}).'
                )

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
        """Converts a single cropped frame pixel coordinate to world XY in mm.

        Args:
            pixel_point: The (x, y) pixel coordinate in the cropped frame.

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
        """Converts a list of cropped frame pixel centroids to world XY in mm.

        Args:
            centroids: List of (x, y) pixel centroid coordinates in cropped frame.

        Returns:
            List of (world_x, world_y) tuples in mm.

        Raises:
            RuntimeError: If homography has not been set up.
        """
        if self._matrix is None:
            raise RuntimeError('Homography not set up. Call setup() first.')
        return [self.transform_point(c) for c in centroids]

    def verify(self) -> None:
        """Displays cropped live frame where clicking prints the world coordinate.

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
            frame = self._camera.get_cropped_frame()
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
            display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
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
    """Detects battery modules using HSV masking and blob detection.

    All processing is performed on the cropped battery work area frame.

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

    def detect(self) -> DetectionResult:
        """Runs the full detection pipeline on the cropped frame.

        Pipeline: get cropped frame -> HSV -> masks -> morphology ->
        blobs -> world coords -> colour classification.

        Warns if more than MAX_MODULES modules are detected, as this
        indicates a calibration or scene error.

        Returns:
            DetectionResult with positions in mm and is_red booleans.
        """
        frame = self._camera.get_cropped_frame()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = self._build_red_mask(hsv)
        blue_mask = self._build_blue_mask(hsv)
        combined_mask = self._clean_mask(cv2.bitwise_or(red_mask, blue_mask))
        centroids = self._get_centroids(combined_mask)

        if len(centroids) > MAX_MODULES:
            print(
                f'WARNING: {len(centroids)} modules detected. '
                f'Maximum is {MAX_MODULES}. Check HSV calibration or scene.'
            )

        world_coords = self._homography.transform_centroids(centroids)
        is_red = self._classify_colours(centroids, red_mask)

        if DEBUG_ENABLED and DEBUG_CONFIG['show_blobs']:
            self._draw_blobs(frame, centroids, world_coords, is_red)

        return DetectionResult(positions=world_coords, is_red=is_red)

    def is_complete(self, result: DetectionResult) -> bool:
        """Returns True if no red modules remain in the detection result.

        Args:
            result: DetectionResult from the last detection pass.

        Returns:
            True if no red modules detected, False otherwise.
        """
        return not any(result.is_red)

    def _build_red_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Builds a binary mask for red pixels from an HSV image.

        Args:
            hsv: HSV image as numpy array.

        Returns:
            Binary red mask as numpy array.
        """
        red_1 = cv2.inRange(hsv, HSV_CONFIG['red_lower_1'], HSV_CONFIG['red_upper_1'])
        red_2 = cv2.inRange(hsv, HSV_CONFIG['red_lower_2'], HSV_CONFIG['red_upper_2'])
        return cv2.bitwise_or(red_1, red_2)

    def _build_blue_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Builds a binary mask for blue pixels from an HSV image.

        Args:
            hsv: HSV image as numpy array.

        Returns:
            Binary blue mask as numpy array.
        """
        return cv2.inRange(hsv, HSV_CONFIG['blue_lower'], HSV_CONFIG['blue_upper'])

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

    def _classify_colours(
        self,
        centroids: list[tuple[float, float]],
        red_mask: np.ndarray,
    ) -> list[bool]:
        """Classifies each blob centroid as red or blue.

        Samples the red mask at each centroid pixel. If the red mask is
        non-zero at that position the module is red, otherwise blue.

        Args:
            centroids: List of (x, y) pixel centroid coordinates.
            red_mask: Binary red mask as numpy array.

        Returns:
            List of booleans. True if red, False if blue.
        """
        return [bool(red_mask[int(py), int(px)]) for px, py in centroids]

    def _draw_blobs(
        self,
        frame: np.ndarray,
        centroids: list[tuple[float, float]],
        world_coords: list[tuple[float, float]],
        is_red: list[bool],
    ) -> None:
        """Draws detected blobs, world coordinates, and colour on the frame.

        Args:
            frame: BGR image to draw on.
            centroids: List of pixel centroid coordinates.
            world_coords: List of world XY coordinates in mm.
            is_red: List of booleans indicating red or blue per module.
        """
        display = frame.copy()
        for (px, py), (wx, wy), red in zip(centroids, world_coords, is_red):
            cx, cy = int(px), int(py)
            colour = (0, 0, 255) if red else (255, 0, 0)  # Red or blue dot.
            cv2.circle(display, (cx, cy), 6, colour, -1)
            cv2.putText(
                display,
                f'({wx:.1f}, {wy:.1f}) {"R" if red else "B"}',
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                2,
            )
        cv2.imshow('Debug: Detected Modules', display)
        cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Calibration class
# ---------------------------------------------------------------------------

class Calibration:
    """Runs interactive calibration procedures for all pipeline parameters.

    Each tuning step shows a controls window with trackbars and a separate
    preview window with the colour frame and binary mask side by side.
    All calibrated values are saved to a JSON file on completion.

    Attributes:
        _camera: Camera instance for frame acquisition.
        _focus_value: Selected focus lock value.
        _crop: Selected crop region parameters.
        _hsv: Selected HSV threshold values.
        _morphology: Selected morphology kernel sizes.
        _blob: Selected blob area limits.
    """

    def __init__(self, camera: Camera) -> None:
        """Initialises Calibration with a Camera instance.

        Args:
            camera: Camera instance for frame acquisition.
        """
        self._camera = camera
        self._focus_value = CAMERA_CONFIG['focus_value']
        self._crop = {
            'x': CAMERA_CONFIG['crop_x'],
            'y': CAMERA_CONFIG['crop_y'],
            'w': CAMERA_CONFIG['crop_width'],
            'h': CAMERA_CONFIG['crop_height'],
        }
        self._hsv = {
            'r1_h_min': int(HSV_CONFIG['red_lower_1'][0]),
            'r1_s_min': int(HSV_CONFIG['red_lower_1'][1]),
            'r1_v_min': int(HSV_CONFIG['red_lower_1'][2]),
            'r1_h_max': int(HSV_CONFIG['red_upper_1'][0]),
            'r2_h_min': int(HSV_CONFIG['red_lower_2'][0]),
            'r2_h_max': int(HSV_CONFIG['red_upper_2'][0]),
            'b_h_min': int(HSV_CONFIG['blue_lower'][0]),
            'b_s_min': int(HSV_CONFIG['blue_lower'][1]),
            'b_v_min': int(HSV_CONFIG['blue_lower'][2]),
            'b_h_max': int(HSV_CONFIG['blue_upper'][0]),
            'b_s_max': int(HSV_CONFIG['blue_upper'][1]),
            'b_v_max': int(HSV_CONFIG['blue_upper'][2]),
        }
        self._morphology = {
            'erode': MORPHOLOGY_CONFIG['erode_kernel_size'],
            'dilate': MORPHOLOGY_CONFIG['dilate_kernel_size'],
        }
        self._blob = {
            'min_area': BLOB_CONFIG['min_area'],
            'max_area': BLOB_CONFIG['max_area'],
        }

    def run(self) -> None:
        """Runs all calibration steps in sequence and saves final config to file."""
        print('\n--- Calibration Mode ---')
        print('Each step shows a live preview. Adjust parameters, then press Q to continue.')
        self.tune_focus()
        self.tune_crop()
        self.tune_hsv()
        self.tune_morphology()
        self.tune_blob()
        self.save_params()
        self._print_config()

    def tune_focus(self) -> None:
        """Opens a live feed with a focus slider to find the correct focus lock value.

        Raises:
            RuntimeError: If a live camera device is not available.
        """
        if self._camera._cap is None:
            print('Focus tuning requires a live camera. Skipping.')
            return
        print('\n[Calibration] Focus tuning. Adjust slider, press Q when done.')
        cv2.namedWindow('Calibration: Focus')
        cv2.createTrackbar('Focus', 'Calibration: Focus', self._focus_value, 255, lambda x: None)
        while cv2.waitKey(1) != ord('q'):
            self._focus_value = cv2.getTrackbarPos('Focus', 'Calibration: Focus')
            self._camera._cap.set(cv2.CAP_PROP_FOCUS, self._focus_value)
            cv2.imshow('Calibration: Focus', self._camera.get_frame())
        cv2.destroyAllWindows()
        print(f'Focus value selected: {self._focus_value}.')

    def tune_crop(self) -> None:
        """Shows full frame. Click top-left then bottom-right corner to set crop region."""
        print('\n[Calibration] Crop tuning. Click top-left then bottom-right. Press Q when done.')
        corners = []

        def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 2:
                corners.append((x, y))
                print(f'Corner {len(corners)} set at ({x}, {y}).')

        cv2.namedWindow('Calibration: Crop')
        cv2.setMouseCallback('Calibration: Crop', on_mouse)
        while cv2.waitKey(1) != ord('q'):
            frame = self._camera.get_frame().copy()
            if len(corners) >= 1:
                cv2.circle(frame, corners[0], 6, (0, 255, 0), -1)
            if len(corners) == 2:
                cv2.rectangle(frame, corners[0], corners[1], (0, 255, 0), 2)
                self._crop['x'] = corners[0][0]
                self._crop['y'] = corners[0][1]
                self._crop['w'] = corners[1][0] - corners[0][0]
                self._crop['h'] = corners[1][1] - corners[0][1]
            cv2.imshow('Calibration: Crop', frame)
        cv2.destroyAllWindows()
        print(f'Crop set: x={self._crop["x"]}, y={self._crop["y"]}, '
              f'w={self._crop["w"]}, h={self._crop["h"]}.')

    def tune_hsv(self) -> None:
        """Runs HSV calibration in three steps: red, blue, combined confirmation."""
        self._tune_red()
        self._tune_blue()
        self._tune_hsv_confirm()

    def tune_morphology(self) -> None:
        """Opens trackbars for erode and dilate kernel sizes with frozen mask preview.

        Uses a frozen frame to prevent flickering from frame-to-frame variation.
        """
        print('\n[Calibration] Morphology tuning. Adjust sliders, press Q when done.')
        ctrl_win = 'Calibration: Morphology Controls'
        preview_win = 'Calibration: Morphology Preview'
        frozen_frame = self._get_crop()  # Frozen to prevent flickering.
        frozen_hsv = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2HSV)
        frozen_mask = cv2.bitwise_or(
            cv2.bitwise_or(
                cv2.inRange(frozen_hsv, HSV_CONFIG['red_lower_1'], HSV_CONFIG['red_upper_1']),
                cv2.inRange(frozen_hsv, HSV_CONFIG['red_lower_2'], HSV_CONFIG['red_upper_2']),
            ),
            cv2.inRange(frozen_hsv, HSV_CONFIG['blue_lower'], HSV_CONFIG['blue_upper']),
        )
        cv2.namedWindow(ctrl_win)
        cv2.namedWindow(preview_win)
        cv2.createTrackbar('Erode', ctrl_win, self._morphology['erode'], 20, lambda x: None)
        cv2.createTrackbar('Dilate', ctrl_win, self._morphology['dilate'], 20, lambda x: None)

        while cv2.waitKey(1) != ord('q'):
            erode_size = max(1, cv2.getTrackbarPos('Erode', ctrl_win))
            dilate_size = max(1, cv2.getTrackbarPos('Dilate', ctrl_win))
            mask = cv2.erode(frozen_mask.copy(), np.ones((erode_size, erode_size), np.uint8))
            mask = cv2.dilate(mask, np.ones((dilate_size, dilate_size), np.uint8))
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow(preview_win, np.hstack([frozen_frame, mask_bgr]))

        self._morphology['erode'] = max(1, cv2.getTrackbarPos('Erode', ctrl_win))
        self._morphology['dilate'] = max(1, cv2.getTrackbarPos('Dilate', ctrl_win))
        cv2.destroyAllWindows()
        print(f'Morphology selected: erode={self._morphology["erode"]}, '
              f'dilate={self._morphology["dilate"]}.')

    def tune_blob(self) -> None:
        """Opens trackbars for blob area limits with frozen blob detection preview.

        Uses a frozen frame to prevent flickering from frame-to-frame variation.
        """
        print('\n[Calibration] Blob tuning. Adjust sliders, press Q when done.')
        ctrl_win = 'Calibration: Blob Controls'
        preview_win = 'Calibration: Blob Preview'
        frozen_frame = self._get_crop()  # Frozen to prevent flickering.
        frozen_hsv = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2HSV)
        frozen_mask = cv2.bitwise_or(
            cv2.bitwise_or(
                cv2.inRange(frozen_hsv, HSV_CONFIG['red_lower_1'], HSV_CONFIG['red_upper_1']),
                cv2.inRange(frozen_hsv, HSV_CONFIG['red_lower_2'], HSV_CONFIG['red_upper_2']),
            ),
            cv2.inRange(frozen_hsv, HSV_CONFIG['blue_lower'], HSV_CONFIG['blue_upper']),
        )
        frozen_mask = cv2.erode(
            frozen_mask,
            np.ones((MORPHOLOGY_CONFIG['erode_kernel_size'],
                     MORPHOLOGY_CONFIG['erode_kernel_size']), np.uint8),
        )
        frozen_mask = cv2.dilate(
            frozen_mask,
            np.ones((MORPHOLOGY_CONFIG['dilate_kernel_size'],
                     MORPHOLOGY_CONFIG['dilate_kernel_size']), np.uint8),
        )
        cv2.namedWindow(ctrl_win)
        cv2.namedWindow(preview_win)
        cv2.createTrackbar('Min Area', ctrl_win, self._blob['min_area'], 5000, lambda x: None)
        cv2.createTrackbar(
            'Max Area', ctrl_win, min(self._blob['max_area'], 50000), 50000, lambda x: None,
        )

        while cv2.waitKey(1) != ord('q'):
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = max(1, cv2.getTrackbarPos('Min Area', ctrl_win))
            params.maxArea = max(1, cv2.getTrackbarPos('Max Area', ctrl_win))
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(cv2.bitwise_not(frozen_mask))
            display = frozen_frame.copy()
            for kp in keypoints:
                cv2.circle(
                    display,
                    (int(kp.pt[0]), int(kp.pt[1])),
                    int(kp.size / 2),
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                display,
                f'Blobs detected: {len(keypoints)}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(preview_win, display)

        self._blob['min_area'] = max(1, cv2.getTrackbarPos('Min Area', ctrl_win))
        self._blob['max_area'] = max(1, cv2.getTrackbarPos('Max Area', ctrl_win))
        cv2.destroyAllWindows()
        print(f'Blob area selected: min={self._blob["min_area"]}, max={self._blob["max_area"]}.')

    def save_params(self, path: str = CALIBRATION_PARAMS_PATH) -> None:
        """Saves all calibrated parameters to a JSON file.

        Args:
            path: Path to save the JSON file. Defaults to CALIBRATION_PARAMS_PATH.
        """
        params = {
            'focus_value': self._focus_value,
            'crop': self._crop,
            'hsv': self._hsv,
            'morphology': self._morphology,
            'blob': self._blob,
        }
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
        print(f'Calibration parameters saved to {path}.')

    @staticmethod
    def load_params(path: str = CALIBRATION_PARAMS_PATH) -> None:
        """Loads calibration parameters from a JSON file into module constants.

        Args:
            path: Path to the JSON file. Defaults to CALIBRATION_PARAMS_PATH.

        Raises:
            FileNotFoundError: If the calibration file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'Calibration file not found: {path}.')
        with open(path) as f:
            params = json.load(f)

        CAMERA_CONFIG['focus_value'] = params['focus_value']
        CAMERA_CONFIG['crop_x'] = params['crop']['x']
        CAMERA_CONFIG['crop_y'] = params['crop']['y']
        CAMERA_CONFIG['crop_width'] = params['crop']['w']
        CAMERA_CONFIG['crop_height'] = params['crop']['h']

        hsv = params['hsv']
        HSV_CONFIG['red_lower_1'] = np.array([hsv['r1_h_min'], hsv['r1_s_min'], hsv['r1_v_min']])
        HSV_CONFIG['red_upper_1'] = np.array([hsv['r1_h_max'], 255, 255])
        HSV_CONFIG['red_lower_2'] = np.array([hsv['r2_h_min'], 100, 100])
        HSV_CONFIG['red_upper_2'] = np.array([hsv['r2_h_max'], 255, 255])
        HSV_CONFIG['blue_lower'] = np.array([hsv['b_h_min'], hsv['b_s_min'], hsv['b_v_min']])
        HSV_CONFIG['blue_upper'] = np.array([hsv['b_h_max'], 255, 255])

        MORPHOLOGY_CONFIG['erode_kernel_size'] = params['morphology']['erode']
        MORPHOLOGY_CONFIG['dilate_kernel_size'] = params['morphology']['dilate']

        BLOB_CONFIG['min_area'] = params['blob']['min_area']
        BLOB_CONFIG['max_area'] = params['blob']['max_area']

        print(f'Calibration parameters loaded from {path}.')

    def _tune_red(self) -> None:
        """Tunes red HSV bounds with live preview. Both red ranges tuned together.

        Controls window shows sliders for Red1 H min/max, Red2 H min/max,
        and shared S min and V min. Preview window shows colour frame left,
        combined red mask right.
        """
        print('\n[Calibration] Red HSV tuning. Adjust sliders, press Q when done.')
        ctrl_win = 'Calibration: Red HSV Controls'
        preview_win = 'Calibration: Red HSV Preview'
        cv2.namedWindow(ctrl_win)
        cv2.namedWindow(preview_win)
        cv2.createTrackbar('Red1 H min', ctrl_win, self._hsv['r1_h_min'], 180, lambda x: None)
        cv2.createTrackbar('Red1 H max', ctrl_win, self._hsv['r1_h_max'], 180, lambda x: None)
        cv2.createTrackbar('Red2 H min', ctrl_win, self._hsv['r2_h_min'], 180, lambda x: None)
        cv2.createTrackbar('Red2 H max', ctrl_win, self._hsv['r2_h_max'], 180, lambda x: None)
        cv2.createTrackbar('S min',      ctrl_win, self._hsv['r1_s_min'], 255, lambda x: None)
        cv2.createTrackbar('V min',      ctrl_win, self._hsv['r1_v_min'], 255, lambda x: None)

        def get(name: str) -> int:
            return cv2.getTrackbarPos(name, ctrl_win)

        while cv2.waitKey(1) != ord('q'):
            frame = self._get_crop()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            s_min = get('S min')
            v_min = get('V min')
            r1 = cv2.inRange(
                hsv,
                np.array([get('Red1 H min'), s_min, v_min]),
                np.array([get('Red1 H max'), 255, 255]),
            )
            r2 = cv2.inRange(
                hsv,
                np.array([get('Red2 H min'), s_min, v_min]),
                np.array([get('Red2 H max'), 255, 255]),
            )
            mask = cv2.bitwise_or(r1, r2)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow(preview_win, np.hstack([frame, mask_bgr]))

        self._hsv['r1_h_min'] = get('Red1 H min')
        self._hsv['r1_h_max'] = get('Red1 H max')
        self._hsv['r2_h_min'] = get('Red2 H min')
        self._hsv['r2_h_max'] = get('Red2 H max')
        self._hsv['r1_s_min'] = get('S min')
        self._hsv['r1_v_min'] = get('V min')
        cv2.destroyAllWindows()
        print('Red HSV values selected.')

    def _tune_blue(self) -> None:
        """Tunes blue HSV bounds with live preview.

        Controls window shows sliders for H min, H max, S min, V min.
        Preview window shows colour frame left, blue mask right.
        """
        print('\n[Calibration] Blue HSV tuning. Adjust sliders, press Q when done.')
        ctrl_win = 'Calibration: Blue HSV Controls'
        preview_win = 'Calibration: Blue HSV Preview'
        cv2.namedWindow(ctrl_win)
        cv2.namedWindow(preview_win)
        cv2.createTrackbar('H min', ctrl_win, self._hsv['b_h_min'], 180, lambda x: None)
        cv2.createTrackbar('H max', ctrl_win, self._hsv['b_h_max'], 180, lambda x: None)
        cv2.createTrackbar('S min', ctrl_win, self._hsv['b_s_min'], 255, lambda x: None)
        cv2.createTrackbar('V min', ctrl_win, self._hsv['b_v_min'], 255, lambda x: None)

        def get(name: str) -> int:
            return cv2.getTrackbarPos(name, ctrl_win)

        while cv2.waitKey(1) != ord('q'):
            frame = self._get_crop()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv,
                np.array([get('H min'), get('S min'), get('V min')]),
                np.array([get('H max'), 255, 255]),
            )
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow(preview_win, np.hstack([frame, mask_bgr]))

        self._hsv['b_h_min'] = get('H min')
        self._hsv['b_h_max'] = get('H max')
        self._hsv['b_s_min'] = get('S min')
        self._hsv['b_v_min'] = get('V min')
        cv2.destroyAllWindows()
        print('Blue HSV values selected.')

    def _tune_hsv_confirm(self) -> None:
        """Shows combined red and blue mask for visual confirmation. Press any key to continue."""
        print('\n[Calibration] Combined HSV confirmation. Press any key to confirm.')
        preview_win = 'Calibration: Combined HSV Preview'
        cv2.namedWindow(preview_win)

        while True:
            frame = self._get_crop()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            s_min = self._hsv['r1_s_min']
            v_min = self._hsv['r1_v_min']
            red_mask = cv2.bitwise_or(
                cv2.inRange(
                    hsv,
                    np.array([self._hsv['r1_h_min'], s_min, v_min]),
                    np.array([self._hsv['r1_h_max'], 255, 255]),
                ),
                cv2.inRange(
                    hsv,
                    np.array([self._hsv['r2_h_min'], s_min, v_min]),
                    np.array([self._hsv['r2_h_max'], 255, 255]),
                ),
            )
            blue_mask = cv2.inRange(
                hsv,
                np.array([self._hsv['b_h_min'], self._hsv['b_s_min'], self._hsv['b_v_min']]),
                np.array([self._hsv['b_h_max'], 255, 255]),
            )
            combined = cv2.bitwise_or(red_mask, blue_mask)
            mask_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            cv2.imshow(preview_win, np.hstack([frame, mask_bgr]))
            if cv2.waitKey(1) != -1:
                break

        cv2.destroyAllWindows()
        print('HSV calibration confirmed.')

    def _get_crop(self) -> np.ndarray:
        """Returns a frame cropped to the currently selected crop region.

        Returns:
            Cropped BGR frame as numpy array.
        """
        frame = self._camera.get_frame()
        x = self._crop['x']
        y = self._crop['y']
        w = self._crop['w']
        h = self._crop['h']
        return frame[y:y + h, x:x + w]

    def _print_config(self) -> None:
        """Prints all calibrated values formatted for copy-pasting into constants."""
        print('\n--- Calibration complete. Values saved to file. For manual update: ---')
        print(f"\nCAMERA_CONFIG['focus_value'] = {self._focus_value}")
        print(f"CAMERA_CONFIG['crop_x'] = {self._crop['x']}")
        print(f"CAMERA_CONFIG['crop_y'] = {self._crop['y']}")
        print(f"CAMERA_CONFIG['crop_width'] = {self._crop['w']}")
        print(f"CAMERA_CONFIG['crop_height'] = {self._crop['h']}")
        print(f"\nHSV_CONFIG['red_lower_1'] = np.array([{self._hsv['r1_h_min']}, "
              f"{self._hsv['r1_s_min']}, {self._hsv['r1_v_min']}])")
        print(f"HSV_CONFIG['red_upper_1'] = np.array([{self._hsv['r1_h_max']}, 255, 255])")
        print(f"HSV_CONFIG['red_lower_2'] = np.array([{self._hsv['r2_h_min']}, "
              f"{self._hsv['r1_s_min']}, {self._hsv['r1_v_min']}])")
        print(f"HSV_CONFIG['red_upper_2'] = np.array([{self._hsv['r2_h_max']}, 255, 255])")
        print(f"HSV_CONFIG['blue_lower'] = np.array([{self._hsv['b_h_min']}, "
              f"{self._hsv['b_s_min']}, {self._hsv['b_v_min']}])")
        print(f"HSV_CONFIG['blue_upper'] = np.array([{self._hsv['b_h_max']}, 255, 255])")
        print(f"\nMORPHOLOGY_CONFIG['erode_kernel_size'] = {self._morphology['erode']}")
        print(f"MORPHOLOGY_CONFIG['dilate_kernel_size'] = {self._morphology['dilate']}")
        print(f"\nBLOB_CONFIG['min_area'] = {self._blob['min_area']}")
        print(f"BLOB_CONFIG['max_area'] = {self._blob['max_area']}")
        print('\n--------------------------------------------------------------------')


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------

def _wait_for_key(message: str = 'Press any key to continue...') -> None:
    """Displays a message on a status window and waits for a keypress.

    Args:
        message: Message to print before waiting.
    """
    print(f'\n{message}')
    status = np.zeros((60, 600, 3), dtype=np.uint8)
    cv2.putText(status, message, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow('Test Status', status)
    while cv2.waitKey(100) == -1:
        pass


# ---------------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('--- Vision Module ---')
    print('Select mode: D = Debug pipeline | C = Calibration')
    status = np.zeros((60, 600, 3), dtype=np.uint8)
    cv2.putText(
        status,
        'Press D for Debug or C for Calibration',
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.imshow('Test Status', status)
    mode = None
    while mode is None:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('d'):
            mode = 'debug'
        elif key == ord('c'):
            mode = 'calibration'
    cv2.destroyAllWindows()

    # Load saved calibration parameters if available.
    if os.path.exists(CALIBRATION_PARAMS_PATH):
        Calibration.load_params()

    camera = Camera.from_device(CAMERA_CONFIG['index'], CAMERA_CONFIG['buffer_size'])

    # --- Calibration mode ---
    if mode == 'calibration':
        calibration = Calibration(camera)
        calibration.run()

    # --- Debug mode ---
    else:
        print('\n--- Debug Mode ---')

        # Step 1: Camera confirmed
        print('\n[1] Camera initialised.')
        _wait_for_key()

        # Step 2: Full frame
        print('\n[2] Full frame...')
        frame = camera.get_frame()
        print(f'Frame shape: {frame.shape}.')
        cv2.imshow('Step 2: Full Frame', frame)
        cv2.waitKey(1)
        _wait_for_key()

        # Step 3: Cropped frame
        print('\n[3] Cropped frame...')
        cropped = camera.get_cropped_frame()
        print(f'Cropped frame shape: {cropped.shape}.')
        cv2.imshow('Step 3: Cropped Frame', cropped)
        cv2.waitKey(1)
        _wait_for_key()

        # Step 4: Homography setup
        print('\n[4] Homography setup...')
        homography = Homography(camera)
        homography.setup()
        _wait_for_key()

        # Step 5: Single point transform
        print('\n[5] Single point transform...')
        test_pixel = (cropped.shape[1] // 2, cropped.shape[0] // 2)  # Cropped frame centre.
        world = homography.transform_point(test_pixel)
        print(f'Pixel {test_pixel} -> World ({world[0]:.1f}, {world[1]:.1f}) mm.')
        _wait_for_key()

        # Step 6: Homography verification
        print('\n[6] Homography verification. Click to test. Press Q to exit.')
        homography.verify()
        _wait_for_key()

        # Step 7: Detector pipeline step by step
        print('\n[7] Detector pipeline step by step...')
        detector = Detector(camera, homography)
        frame = camera.get_cropped_frame()

        print('  [7a] HSV conversion...')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('Step 7a: HSV', hsv)
        cv2.waitKey(1)
        _wait_for_key()

        print('  [7b] Red mask...')
        red_mask = detector._build_red_mask(hsv)
        cv2.imshow('Step 7b: Red Mask', red_mask)
        cv2.waitKey(1)
        _wait_for_key()

        print('  [7c] Blue mask...')
        blue_mask = detector._build_blue_mask(hsv)
        cv2.imshow('Step 7c: Blue Mask', blue_mask)
        cv2.waitKey(1)
        _wait_for_key()

        print('  [7d] Combined mask...')
        combined_mask = cv2.bitwise_or(red_mask, blue_mask)
        cv2.imshow('Step 7d: Combined Mask', combined_mask)
        cv2.waitKey(1)
        _wait_for_key()

        print('  [7e] Morphology cleanup...')
        mask_clean = detector._clean_mask(combined_mask)
        cv2.imshow('Step 7e: Clean Mask', mask_clean)
        cv2.waitKey(1)
        _wait_for_key()

        print('  [7f] Blob detection...')
        centroids = detector._get_centroids(mask_clean)
        print(f'  Detected {len(centroids)} blob(s).')
        for i, (px, py) in enumerate(centroids):
            print(f'    Blob {i + 1}: Pixel ({px:.1f}, {py:.1f}).')
        _wait_for_key()

        print('  [7g] Colour classification...')
        is_red = detector._classify_colours(centroids, red_mask)
        for i, red in enumerate(is_red):
            print(f'    Module {i + 1}: {"Red" if red else "Blue"}.')
        _wait_for_key()

        print('  [7h] World coordinate transform...')
        world_coords = homography.transform_centroids(centroids)
        for i, ((wx, wy), red) in enumerate(zip(world_coords, is_red)):
            print(f'    Module {i + 1}: World ({wx:.1f}, {wy:.1f}) mm — {"Red" if red else "Blue"}.')
        detector._draw_blobs(frame, centroids, world_coords, is_red)
        cv2.waitKey(1)
        _wait_for_key()

        # Step 8: Full detect() call
        print('\n[8] Full detect() returning DetectionResult...')
        result = detector.detect()
        print(f'Detected {len(result.positions)} module(s).')
        for i, ((wx, wy), red) in enumerate(zip(result.positions, result.is_red)):
            print(f'  Module {i + 1}: ({wx:.1f}, {wy:.1f}) mm — {"Red" if red else "Blue"}.')
        print(f'Job complete: {detector.is_complete(result)}.')
        _wait_for_key()

        # Step 9: Offline mode
        print('\n[9] Offline mode with static image...')
        cv2.imwrite('test_frame.jpg', frame)
        offline_camera = Camera.from_image('test_frame.jpg')
        offline_homography = Homography(offline_camera)
        offline_detector = Detector(offline_camera, offline_homography)
        offline_result = offline_detector.detect()
        print(f'Offline detected {len(offline_result.positions)} module(s).')
        for i, ((wx, wy), red) in enumerate(
            zip(offline_result.positions, offline_result.is_red)
        ):
            print(f'  Module {i + 1}: ({wx:.1f}, {wy:.1f}) mm — {"Red" if red else "Blue"}.')
        print(f'Job complete: {offline_detector.is_complete(offline_result)}.')
        _wait_for_key()

    camera.release()
    cv2.destroyAllWindows()
    print('\n--- Done ---')