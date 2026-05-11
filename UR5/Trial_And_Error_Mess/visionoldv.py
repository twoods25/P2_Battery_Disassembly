"""Vision Module"""


# Import of Libraries
import cv2  # Computer Vision Library
import numpy as np  # Numpy Library
import os  # Interact with OS


# Constants
DEBUG_ENABLED = True
DEBUG_CONFIG = {
    'log_initialize': True,  # Logs camera index, buffer, resolution, focus on init
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
    'crop_x': 705,  # Top-left x of battery box region
    'crop_y': 270,  # Top-left y of battery box region
    'crop_width': 800,  # Width of crop region
    'crop_height': 600,  # Height of crop region
    'display_scale': 1.0,
}
CAMERA_POSITION_XYZ = np.array([520, 160, 982])
CORNER_LABELS = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
HOMOGRAPHY_SAVE_PATH = 'homography.npy'

class Camera:
    """Represents a camera device or static image source for frame acquisition.

    Attributes:
        _cap: The cv2.VideoCapture instance. None if using test image.
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

    @classmethod
    def from_image(cls, image_path: str) -> 'Camera':
        """Creates a Camera instance from a static test image.

        Args:
            image_path: Path to the image file.

        Returns:
            A Camera instance loaded with the static image.

        Raises:
            ValueError: If the image could not be loaded.
        """
    def get_frame(self) -> np.ndarray:
        """Retrieves a fresh frame or test image.

        Returns:
            A BGR image as a numpy array of shape (height, width, 3).

        Raises:
            RuntimeError: If the camera has not been initialized.
            RuntimeError: If the frame could not be retrieved.
        """

    def crop_to_battery(self, frame: np.ndarray) -> np.ndarray:
        """Crops frame to the fixed battery box work area.

        Args:
            frame: Full resolution camera frame.

        Returns:
            Cropped frame containing only the battery box region.
        """

    def show_live_feed(self) -> None:
        """Displays a live camera feed with a center crosshair until any key is pressed.

        Raises:
            RuntimeError: If the camera has not been initialized.
        """

    def release(self) -> None:
        """Releases the camera resource cleanly."""


"""
Module testing
1. cv2.VideoCapture → get frame
2. hardcode camera_xyz
3. get/estimate focal length
4. cv2.imshow + cv2.setMouseCallback → collect 4 clicks
5. convert clicks to world coords using camera xyz + focal length
6. cv2.getPerspectiveTransform → H
"""

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