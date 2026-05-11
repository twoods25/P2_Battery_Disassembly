"""UR5 vision module for EV battery disassembly robot.

Loads calibration parameters from file and provides camera acquisition,
homography, and blob detection for identifying and localising battery modules.

Usage:
    Calibration.load_params()
    camera = Camera.from_device(CAMERA_CONFIG['index'], CAMERA_CONFIG['buffer_size'])
    homography = Homography(camera)
    homography.setup()
    detector = Detector(camera, homography)
    result = detector.detect()
"""

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration — populated by load_params() at startup
# ---------------------------------------------------------------------------

CAMERA_CONFIG = {
    'index': 1,
    'buffer_size': 1,
    'focus_value': 0,
    'frame_width': 1920,
    'frame_height': 1080,
    'crop_x': 705,
    'crop_y': 270,
    'crop_width': 800,
    'crop_height': 600,
}

HSV_CONFIG = {
    'red_lower_1': np.array([0, 100, 100]),
    'red_upper_1': np.array([10, 255, 255]),
    'red_lower_2': np.array([170, 100, 100]),
    'red_upper_2': np.array([180, 255, 255]),
    'blue_lower': np.array([100, 100, 50]),
    'blue_upper': np.array([130, 255, 255]),
}

MORPHOLOGY_CONFIG = {
    'erode_kernel_size': 5,
    'dilate_kernel_size': 8,
}

BLOB_CONFIG = {
    'min_area': 500,
    'max_area': 50000,
}

# World XY coordinates of the 4 calibration targets in mm.
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
CALIBRATION_TARGETS_WORLD = np.array([
    [480, 32],
    [480, 320],
    [640, 320],
    [640, 32],
], dtype=np.float32)

CALIBRATION_PARAMS_PATH = 'calibration_params.json'
HOMOGRAPHY_SAVE_PATH = 'homography.npy'
CORNER_LABELS = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
MAX_MODULES = 8
BAR_HEIGHT = 60  # Height of instruction bar above setup images in pixels.


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
        print(f'Camera opened on index {camera_index}.')
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
        return frame

    def get_cropped_frame(self) -> np.ndarray:
        """Retrieves a frame cropped to the battery work area.

        Returns:
            A BGR image cropped to the battery work area.
        """
        frame = self.get_frame()
        x = CAMERA_CONFIG['crop_x']
        y = CAMERA_CONFIG['crop_y']
        w = CAMERA_CONFIG['crop_width']
        h = CAMERA_CONFIG['crop_height']
        return frame[y:y + h, x:x + w]

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
        matrix to file after calibration.

        Raises:
            RuntimeError: If homography setup is cancelled by the user.
        """
        if self._matrix is not None:
            frame = self._camera.get_cropped_frame()
            bar = np.zeros((60, frame.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                bar, 'Saved homography found.',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            cv2.putText(
                bar, 'L = load previous   C = recalibrate   Q = quit',
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1,
            )
            cv2.imshow('Homography Setup', np.vstack([bar, frame]))
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
                mouse_pos[0] = (x, y - BAR_HEIGHT)  # Adjust for instruction bar offset.
            elif event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append((x, y - BAR_HEIGHT))  # Adjust for instruction bar offset.
                print(
                    f'Click {len(clicks)}: {CORNER_LABELS[len(clicks) - 1]}'
                    f' at ({x}, {y - BAR_HEIGHT}).'
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

        # Instruction text in a clean two-line black bar above the image.
        if len(clicks) < 4:
            line1 = f'Click {CORNER_LABELS[len(clicks)]}  ({len(clicks)}/4 selected)'
            line2 = 'U = undo last click   Q = quit'
        else:
            line1 = 'All 4 points selected.'
            line2 = 'Press any key to confirm   U = undo last click'
        bar = np.zeros((60, display.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bar, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        return np.vstack([bar, display])

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

        Warns if more than MAX_MODULES modules are detected.

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

        Centroids are sorted top-left to bottom-right so module numbers
        are consistent regardless of detection order. Modules 1-4 start
        from the top-left, modules 5-8 from the bottom-left.

        Args:
            mask: Binary mask as numpy array.

        Returns:
            List of (x, y) pixel centroid coordinates sorted by position.
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
        centroids = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        row_height = CAMERA_CONFIG['crop_height'] // 5  # Threshold smaller than row spacing.
        return sorted(centroids, key=lambda pt: (int(pt[1] / row_height), pt[0]))  # Row then column.

    def _classify_colours(
        self,
        centroids: list[tuple[float, float]],
        red_mask: np.ndarray,
    ) -> list[bool]:
        """Classifies each blob centroid as red or blue.

        Args:
            centroids: List of (x, y) pixel centroid coordinates.
            red_mask: Binary red mask as numpy array.

        Returns:
            List of booleans. True if red, False if blue.
        """
        return [bool(red_mask[int(py), int(px)]) for px, py in centroids]


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------

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
    HSV_CONFIG['red_lower_2'] = np.array([hsv['r2_h_min'], hsv['r1_s_min'], hsv['r1_v_min']])
    HSV_CONFIG['red_upper_2'] = np.array([hsv['r2_h_max'], 255, 255])
    HSV_CONFIG['blue_lower'] = np.array([hsv['b_h_min'], hsv['b_s_min'], hsv['b_v_min']])
    HSV_CONFIG['blue_upper'] = np.array([hsv['b_h_max'], 255, 255])

    MORPHOLOGY_CONFIG['erode_kernel_size'] = params['morphology']['erode']
    MORPHOLOGY_CONFIG['dilate_kernel_size'] = params['morphology']['dilate']

    BLOB_CONFIG['min_area'] = params['blob']['min_area']
    BLOB_CONFIG['max_area'] = params['blob']['max_area']

    print(f'Calibration parameters loaded from {path}.')


# ---------------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('--- UR5 Vision Module Test ---')

    # Load calibration parameters.
    load_params()

    # Initialise camera.
    camera = Camera.from_device(CAMERA_CONFIG['index'], CAMERA_CONFIG['buffer_size'])

    # Homography setup — ask to load or recalibrate.
    homography = Homography(camera)
    homography.setup()

    # Run detection and display result.
    detector = Detector(camera, homography)
    result = detector.detect()

    print(f'\nDetected {len(result.positions)} module(s):')
    for i, ((wx, wy), red) in enumerate(zip(result.positions, result.is_red)):
        print(f'  Module {i + 1}: ({wx:.1f}, {wy:.1f}) mm — {"Red" if red else "Blue"}.')
    print(f'is_red array: {result.is_red}')
    print(f'positions array: {result.positions}')

    # Display blob detection result — number on image, info bar below.
    frame = camera.get_cropped_frame()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = detector._build_red_mask(hsv)
    blue_mask = detector._build_blue_mask(hsv)
    combined_mask = detector._clean_mask(cv2.bitwise_or(red_mask, blue_mask))
    centroids = detector._get_centroids(combined_mask)
    display = frame.copy()

    for i, ((px, py), (wx, wy), red) in enumerate(
        zip(centroids, result.positions, result.is_red)
    ):
        cx, cy = int(px), int(py)
        colour = (0, 0, 255) if red else (255, 0, 0)
        cv2.circle(display, (cx, cy), 20, (0, 255, 0), 2)   # Green circle outline.
        cv2.circle(display, (cx, cy), 4, colour, -1)         # Red or blue centre dot.
        cv2.putText(                                          # Module number only on image.
            display,
            str(i + 1),
            (cx - 6, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Info bar below image — one entry per module.
    bar_height = 30 + len(result.positions) * 22
    bar = np.zeros((bar_height, display.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, 'Module  Colour  World X   World Y', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    for i, ((wx, wy), red) in enumerate(zip(result.positions, result.is_red)):
        colour = (0, 0, 255) if red else (255, 0, 0)
        cv2.putText(
            bar,
            f'  {i + 1}       {"Red " if red else "Blue"}    {wx:6.1f}    {wy:6.1f}',
            (10, 42 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
        )

    cv2.imshow('Detection Result — Press any key to close', np.vstack([display, bar]))
    cv2.waitKey(0)

    camera.release()
    cv2.destroyAllWindows()
    print('\n--- Done ---')