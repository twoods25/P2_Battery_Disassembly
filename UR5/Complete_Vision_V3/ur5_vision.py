"""UR5 vision module for EV battery disassembly robot.

Pure data pipeline. No UI, no display, no user interaction.
Can be used standalone by the robot without any other module.

Usage:
    from ur5_vision import load_params, Camera, Homography, Detector

    load_params()
    camera = Camera.from_device(CAMERA_CONFIG['index'], CAMERA_CONFIG['buffer_size'])
    homography = Homography(camera)
    homography.compute(pixel_points)
    detector = Detector(camera, homography)
    result = detector.detect()
    # result.positions  -> list of (world_x, world_y) in mm
    # result.is_red     -> list of booleans, True = red module
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
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
# Update if targets are moved.
CALIBRATION_TARGETS_WORLD = np.array([
    [480, 80],
    [480, 320],
    [640, 320],
    [640, 80],
], dtype=np.float32)

CALIBRATION_PARAMS_PATH = 'calibration_params.json'
HOMOGRAPHY_SAVE_PATH = 'homography.npy'
CORNER_LABELS = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
MAX_MODULES = 8  # Maximum expected modules in a battery pack.


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Structured output from a single detection pass.

    A dataclass is used here because DetectionResult is a pure data
    container with no behaviour — it just holds the three related outputs
    of one detection pass together in a named, self-documenting structure.

    Attributes:
        positions: List of (world_x, world_y) tuples in mm.
        is_red: List of booleans. True if the module is red, False if blue.
        pixel_centroids: List of (x, y) pixel positions in cropped frame coordinates.
    """

    positions: list
    is_red: list
    pixel_centroids: list


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    """Manages camera hardware or static image source for frame acquisition.

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
        camera._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)             # Locks focus for stable image.
        camera._cap.set(cv2.CAP_PROP_FOCUS, CAMERA_CONFIG['focus_value'])
        camera._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['frame_width'])
        camera._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['frame_height'])
        for _ in range(5):  # Warmup frames to allow camera to initialise.
            camera._cap.grab()
        return camera

    @classmethod
    def from_image(cls, image_path: str) -> 'Camera':
        """Creates a Camera instance from a static image for offline use.

        Supported formats: jpg, png, bmp. File must be in the same folder
        as the script, or provide a full path. Example: 'snapshot.jpg'

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
        """Retrieves a fresh full frame from the camera or returns the static image.

        Returns:
            A BGR image as a numpy array of shape (height, width, 3).

        Raises:
            RuntimeError: If neither camera nor static image is available.
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
# Homography
# ---------------------------------------------------------------------------

class Homography:
    """Manages perspective transform between pixel and world coordinates.

    Attributes:
        _matrix: The 3x3 homography matrix H. None until compute() is called.
    """

    def __init__(self) -> None:
        """Initialises Homography and loads saved matrix from file if available."""
        self._matrix = None
        if os.path.exists(HOMOGRAPHY_SAVE_PATH):
            self._matrix = np.load(HOMOGRAPHY_SAVE_PATH)

    def compute(self, pixel_points: np.ndarray) -> None:
        """Computes and saves the homography matrix from 4 pixel-to-world correspondences.

        Args:
            pixel_points: numpy array of shape (4, 2) with pixel coordinates
                in the cropped frame. Order must match CALIBRATION_TARGETS_WORLD:
                Top-Left, Top-Right, Bottom-Right, Bottom-Left.

        Raises:
            ValueError: If pixel_points does not have shape (4, 2).
        """
        if pixel_points.shape != (4, 2):
            raise ValueError(f'Expected pixel_points shape (4, 2), got {pixel_points.shape}.')
        self._matrix = cv2.getPerspectiveTransform(
            pixel_points.astype(np.float32),
            CALIBRATION_TARGETS_WORLD,
        )
        np.save(HOMOGRAPHY_SAVE_PATH, self._matrix)

    def transform_point(self, pixel_point: tuple[int, int]) -> tuple[float, float]:
        """Converts a single cropped frame pixel coordinate to world XY in mm.

        Args:
            pixel_point: The (x, y) pixel coordinate in the cropped frame.

        Returns:
            A tuple of (world_x, world_y) in mm.

        Raises:
            RuntimeError: If homography has not been computed.
        """
        if self._matrix is None:
            raise RuntimeError('Homography not computed. Call compute() first.')
        point = np.array([[[float(pixel_point[0]), float(pixel_point[1])]]])
        result = cv2.perspectiveTransform(point, self._matrix)
        return float(result[0][0][0]), float(result[0][0][1])

    def transform_centroids(
        self,
        centroids: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Converts a list of pixel centroids to world XY coordinates in mm.

        Args:
            centroids: List of (x, y) pixel coordinates in the cropped frame.

        Returns:
            List of (world_x, world_y) tuples in mm.

        Raises:
            RuntimeError: If homography has not been computed.
        """
        if self._matrix is None:
            raise RuntimeError('Homography not computed. Call compute() first.')
        return [self.transform_point(c) for c in centroids]

    @property
    def is_ready(self) -> bool:
        """Returns True if the homography matrix has been computed or loaded."""
        return self._matrix is not None


# ---------------------------------------------------------------------------
# Detector
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

        Logs a warning if more than MAX_MODULES modules are detected,
        as this indicates a calibration or scene error.

        Returns:
            DetectionResult with positions, is_red, and pixel_centroids.
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
        return DetectionResult(
            positions=world_coords,
            is_red=is_red,
            pixel_centroids=centroids,
        )

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
        """Detects blobs in the mask and returns sorted pixel centroids.

        Sorted top-left to bottom-right so module numbers are consistent
        regardless of detection order: 1-4 top half, 5-8 bottom half.

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
        return sorted(centroids, key=lambda pt: (int(pt[1] / row_height), pt[0]))

    def _classify_colours(
        self,
        centroids: list[tuple[float, float]],
        red_mask: np.ndarray,
    ) -> list[bool]:
        """Classifies each blob centroid as red or blue.

        Samples the red mask at the centroid pixel. Non-zero means red.

        Args:
            centroids: List of (x, y) pixel centroid coordinates.
            red_mask: Binary red mask as numpy array.

        Returns:
            List of booleans. True if red, False if blue.
        """
        return [bool(red_mask[int(py), int(px)]) for px, py in centroids]


# ---------------------------------------------------------------------------
# Parameter loader
# ---------------------------------------------------------------------------

def load_params(path: str = CALIBRATION_PARAMS_PATH) -> None:
    """Loads calibration parameters from JSON into module constants.

    If the file does not exist, saves and uses the current default values
    so the program can run without requiring calibration first.

    Must be called before Camera.from_device() so focus and resolution
    settings are applied correctly on camera initialisation.

    Args:
        path: Path to the JSON file. Defaults to CALIBRATION_PARAMS_PATH.
    """
    if not os.path.exists(path):
        print(f'No calibration file found at {path}. Using and saving defaults.')
        _save_default_params(path)
        return

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


def _save_default_params(path: str) -> None:
    """Saves current module constants as default calibration JSON.

    Args:
        path: Path to write the JSON file.
    """
    params = {
        'focus_value': CAMERA_CONFIG['focus_value'],
        'crop': {
            'x': CAMERA_CONFIG['crop_x'],
            'y': CAMERA_CONFIG['crop_y'],
            'w': CAMERA_CONFIG['crop_width'],
            'h': CAMERA_CONFIG['crop_height'],
        },
        'hsv': {
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
        },
        'morphology': {
            'erode': MORPHOLOGY_CONFIG['erode_kernel_size'],
            'dilate': MORPHOLOGY_CONFIG['dilate_kernel_size'],
        },
        'blob': {
            'min_area': BLOB_CONFIG['min_area'],
            'max_area': BLOB_CONFIG['max_area'],
        },
    }
    with open(path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f'Default calibration parameters saved to {path}.')


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('--- ur5_vision standalone test ---')

    # Load calibration parameters.
    load_params()
    print('Parameters loaded.')

    # --- Camera ---
    # To test with a static image instead of a live camera, replace the next
    # line with: camera = Camera.from_image('snapshot.jpg')
    # The image must be in the same folder. Supported formats: jpg, png, bmp.
    camera = Camera.from_device(CAMERA_CONFIG['index'], CAMERA_CONFIG['buffer_size'])
    print(f'Full frame shape:    {camera.get_frame().shape}')
    print(f'Cropped frame shape: {camera.get_cropped_frame().shape}')

    # --- Homography ---
    # Provide 4 pixel clicks here to test compute() without UI.
    # These are example coordinates — replace with real measured clicks.
    homography = Homography()
    if not homography.is_ready:
        test_clicks = np.array([[100, 80], [300, 80], [300, 220], [100, 220]], dtype=np.float32)
        homography.compute(test_clicks)
        print('Homography computed from test clicks.')
    else:
        print('Homography loaded from file.')

    test_pixel = (200, 150)
    wx, wy = homography.transform_point(test_pixel)
    print(f'Transform test: pixel {test_pixel} -> world ({wx:.1f}, {wy:.1f}) mm.')

    # --- Detector ---
    detector = Detector(camera, homography)
    result = detector.detect()

    print(f'\nDetected {len(result.positions)} module(s):')
    for i, ((px, py), (wx, wy), red) in enumerate(
        zip(result.pixel_centroids, result.positions, result.is_red)
    ):
        print(
            f'  Module {i + 1}: pixel ({px:.0f}, {py:.0f})'
            f' -> world ({wx:.1f}, {wy:.1f}) mm'
            f' — {"Red" if red else "Blue"}.'
        )
    print(f'\npositions: {result.positions}')
    print(f'is_red:    {result.is_red}')

    camera.release()
    print('\n--- Done ---')
