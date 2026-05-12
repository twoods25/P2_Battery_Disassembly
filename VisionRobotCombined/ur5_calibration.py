"""UR5 calibration tool for EV battery disassembly robot.

Standalone tool for tuning all vision pipeline parameters and saving
them to calibration_params.json. Run this file directly to calibrate.

Does not need to be imported by ur5_main.py. To trigger calibration
from main on startup, see the commented line in ur5_main.py.

Usage:
    python ur5_calibration.py
"""

import json

import cv2
import numpy as np

from ur5_vision import (
    BLOB_CONFIG,
    CALIBRATION_PARAMS_PATH,
    CAMERA_CONFIG,
    HSV_CONFIG,
    MORPHOLOGY_CONFIG,
    Camera,
)


class Calibration:
    """Runs interactive calibration for all pipeline parameters.

    Each step shows a live preview window and a separate controls window
    with trackbars. Completed values are saved to calibration_params.json.

    Attributes:
        _camera: Camera instance for frame acquisition.
        _focus_value: Selected focus lock value.
        _crop: Selected crop region.
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
        """Runs all calibration steps in sequence and saves results to file."""
        print('--- Calibration ---')
        print('Each step: adjust parameters, press Q to continue.')
        self._tune_focus()
        self._tune_crop()
        self._tune_red_hsv()
        self._tune_blue_hsv()
        self._confirm_hsv()
        self._tune_morphology()
        self._tune_blob()
        self._save()

    def _tune_focus(self) -> None:
        """Trackbar slider for focus lock value with live camera preview."""
        if self._camera._cap is None:
            print('Focus tuning requires a live camera. Skipping.')
            return
        print('\n[1/7] Focus tuning. Adjust slider, press Q when done.')
        cv2.namedWindow('Calibration: Focus')
        cv2.createTrackbar('Focus', 'Calibration: Focus', self._focus_value, 255, lambda x: None)
        while cv2.waitKey(1) != ord('q'):
            self._focus_value = cv2.getTrackbarPos('Focus', 'Calibration: Focus')
            self._camera._cap.set(cv2.CAP_PROP_FOCUS, self._focus_value)
            cv2.imshow('Calibration: Focus', self._camera.get_frame())
        cv2.destroyAllWindows()
        print(f'Focus: {self._focus_value}.')

    def _tune_crop(self) -> None:
        """Click top-left then bottom-right on full frame to set crop region."""
        print('\n[2/7] Crop tuning. Click top-left then bottom-right corner. Press Q when done.')
        corners = []

        def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 2:
                corners.append((x, y))
                print(f'Corner {len(corners)} at ({x}, {y}).')

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
        print(f'Crop: x={self._crop["x"]}, y={self._crop["y"]}, '
              f'w={self._crop["w"]}, h={self._crop["h"]}.')

    def _tune_red_hsv(self) -> None:
        """Trackbars for red HSV bounds. Both red ranges tuned together."""
        print('\n[3/7] Red HSV tuning. Adjust sliders, press Q when done.')
        ctrl = 'Calibration: Red HSV Controls'
        prev = 'Calibration: Red HSV Preview'
        cv2.namedWindow(ctrl)
        cv2.namedWindow(prev)
        cv2.createTrackbar('Red1 H min', ctrl, self._hsv['r1_h_min'], 180, lambda x: None)
        cv2.createTrackbar('Red1 H max', ctrl, self._hsv['r1_h_max'], 180, lambda x: None)
        cv2.createTrackbar('Red2 H min', ctrl, self._hsv['r2_h_min'], 180, lambda x: None)
        cv2.createTrackbar('Red2 H max', ctrl, self._hsv['r2_h_max'], 180, lambda x: None)
        cv2.createTrackbar('S min',      ctrl, self._hsv['r1_s_min'], 255, lambda x: None)
        cv2.createTrackbar('V min',      ctrl, self._hsv['r1_v_min'], 255, lambda x: None)

        def get(name: str) -> int:
            return cv2.getTrackbarPos(name, ctrl)

        while cv2.waitKey(1) != ord('q'):
            frame = self._get_crop()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            s, v = get('S min'), get('V min')
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([get('Red1 H min'), s, v]),
                            np.array([get('Red1 H max'), 255, 255])),
                cv2.inRange(hsv, np.array([get('Red2 H min'), s, v]),
                            np.array([get('Red2 H max'), 255, 255])),
            )
            cv2.imshow(prev, np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

        self._hsv.update({
            'r1_h_min': get('Red1 H min'), 'r1_h_max': get('Red1 H max'),
            'r2_h_min': get('Red2 H min'), 'r2_h_max': get('Red2 H max'),
            'r1_s_min': get('S min'),      'r1_v_min': get('V min'),
        })
        cv2.destroyAllWindows()
        print('Red HSV done.')

    def _tune_blue_hsv(self) -> None:
        """Trackbars for blue HSV bounds with live mask preview."""
        print('\n[4/7] Blue HSV tuning. Adjust sliders, press Q when done.')
        ctrl = 'Calibration: Blue HSV Controls'
        prev = 'Calibration: Blue HSV Preview'
        cv2.namedWindow(ctrl)
        cv2.namedWindow(prev)
        cv2.createTrackbar('H min', ctrl, self._hsv['b_h_min'], 180, lambda x: None)
        cv2.createTrackbar('H max', ctrl, self._hsv['b_h_max'], 180, lambda x: None)
        cv2.createTrackbar('S min', ctrl, self._hsv['b_s_min'], 255, lambda x: None)
        cv2.createTrackbar('V min', ctrl, self._hsv['b_v_min'], 255, lambda x: None)

        def get(name: str) -> int:
            return cv2.getTrackbarPos(name, ctrl)

        while cv2.waitKey(1) != ord('q'):
            frame = self._get_crop()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv,
                np.array([get('H min'), get('S min'), get('V min')]),
                np.array([get('H max'), 255, 255]),
            )
            cv2.imshow(prev, np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

        self._hsv.update({
            'b_h_min': get('H min'), 'b_h_max': get('H max'),
            'b_s_min': get('S min'), 'b_v_min': get('V min'),
        })
        cv2.destroyAllWindows()
        print('Blue HSV done.')

    def _confirm_hsv(self) -> None:
        """Shows combined red and blue mask for confirmation. Press any key to continue."""
        print('\n[5/7] Combined HSV preview. Press any key to confirm.')
        while True:
            frame = self._get_crop()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            s, v = self._hsv['r1_s_min'], self._hsv['r1_v_min']
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([self._hsv['r1_h_min'], s, v]),
                            np.array([self._hsv['r1_h_max'], 255, 255])),
                cv2.inRange(hsv, np.array([self._hsv['r2_h_min'], s, v]),
                            np.array([self._hsv['r2_h_max'], 255, 255])),
            )
            blue_mask = cv2.inRange(
                hsv,
                np.array([self._hsv['b_h_min'], self._hsv['b_s_min'], self._hsv['b_v_min']]),
                np.array([self._hsv['b_h_max'], 255, 255]),
            )
            combined = cv2.bitwise_or(red_mask, blue_mask)
            cv2.imshow(
                'Calibration: Combined HSV Preview',
                np.hstack([frame, cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)]),
            )
            if cv2.waitKey(1) != -1:
                break
        cv2.destroyAllWindows()
        print('HSV confirmed.')

    def _tune_morphology(self) -> None:
        """Trackbars for erosion and dilation kernel sizes. Frozen frame to prevent flicker."""
        print('\n[6/7] Morphology tuning. Adjust sliders, press Q when done.')
        ctrl = 'Calibration: Morphology Controls'
        prev = 'Calibration: Morphology Preview'
        frozen_frame = self._get_crop()
        frozen_mask = self._build_mask(frozen_frame)
        cv2.namedWindow(ctrl)
        cv2.namedWindow(prev)
        cv2.createTrackbar('Erode',  ctrl, self._morphology['erode'],  20, lambda x: None)
        cv2.createTrackbar('Dilate', ctrl, self._morphology['dilate'], 20, lambda x: None)

        while cv2.waitKey(1) != ord('q'):
            e = max(1, cv2.getTrackbarPos('Erode', ctrl))
            d = max(1, cv2.getTrackbarPos('Dilate', ctrl))
            mask = cv2.erode(frozen_mask.copy(), np.ones((e, e), np.uint8))
            mask = cv2.dilate(mask, np.ones((d, d), np.uint8))
            cv2.imshow(prev, np.hstack([frozen_frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

        self._morphology['erode'] = max(1, cv2.getTrackbarPos('Erode', ctrl))
        self._morphology['dilate'] = max(1, cv2.getTrackbarPos('Dilate', ctrl))
        cv2.destroyAllWindows()
        print(f'Morphology: erode={self._morphology["erode"]}, dilate={self._morphology["dilate"]}.')

    def _tune_blob(self) -> None:
        """Trackbars for blob area limits. Frozen frame to prevent flicker."""
        print('\n[7/7] Blob tuning. Adjust sliders, press Q when done.')
        ctrl = 'Calibration: Blob Controls'
        prev = 'Calibration: Blob Preview'
        frozen_frame = self._get_crop()
        frozen_mask = self._build_mask(frozen_frame)
        frozen_mask = cv2.erode(
            frozen_mask,
            np.ones((self._morphology['erode'], self._morphology['erode']), np.uint8),
        )
        frozen_mask = cv2.dilate(
            frozen_mask,
            np.ones((self._morphology['dilate'], self._morphology['dilate']), np.uint8),
        )
        cv2.namedWindow(ctrl)
        cv2.namedWindow(prev)
        cv2.createTrackbar('Min Area', ctrl, self._blob['min_area'], 5000,  lambda x: None)
        cv2.createTrackbar('Max Area', ctrl, min(self._blob['max_area'], 50000), 50000, lambda x: None)

        while cv2.waitKey(1) != ord('q'):
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = max(1, cv2.getTrackbarPos('Min Area', ctrl))
            params.maxArea = max(1, cv2.getTrackbarPos('Max Area', ctrl))
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(cv2.bitwise_not(frozen_mask))
            display = frozen_frame.copy()
            for kp in keypoints:
                cv2.circle(display, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), (0, 255, 0), 2)
            cv2.putText(
                display, f'Blobs: {len(keypoints)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow(prev, display)

        self._blob['min_area'] = max(1, cv2.getTrackbarPos('Min Area', ctrl))
        self._blob['max_area'] = max(1, cv2.getTrackbarPos('Max Area', ctrl))
        cv2.destroyAllWindows()
        print(f'Blob: min={self._blob["min_area"]}, max={self._blob["max_area"]}.')

    def _build_mask(self, frame: np.ndarray) -> np.ndarray:
        """Builds combined red and blue binary mask from a BGR frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Combined binary mask as numpy array.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s, v = self._hsv['r1_s_min'], self._hsv['r1_v_min']
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([self._hsv['r1_h_min'], s, v]),
                        np.array([self._hsv['r1_h_max'], 255, 255])),
            cv2.inRange(hsv, np.array([self._hsv['r2_h_min'], s, v]),
                        np.array([self._hsv['r2_h_max'], 255, 255])),
        )
        blue_mask = cv2.inRange(
            hsv,
            np.array([self._hsv['b_h_min'], self._hsv['b_s_min'], self._hsv['b_v_min']]),
            np.array([self._hsv['b_h_max'], 255, 255]),
        )
        return cv2.bitwise_or(red_mask, blue_mask)

    def _get_crop(self) -> np.ndarray:
        """Returns a frame cropped to the currently selected crop region.

        Returns:
            Cropped BGR frame as numpy array.
        """
        frame = self._camera.get_frame()
        x, y, w, h = self._crop['x'], self._crop['y'], self._crop['w'], self._crop['h']
        return frame[y:y + h, x:x + w]

    def _save(self) -> None:
        """Saves all calibrated values to calibration_params.json."""
        params = {
            'focus_value': self._focus_value,
            'crop': self._crop,
            'hsv': self._hsv,
            'morphology': self._morphology,
            'blob': self._blob,
        }
        with open(CALIBRATION_PARAMS_PATH, 'w') as f:
            json.dump(params, f, indent=4)
        print(f'\nCalibration saved to {CALIBRATION_PARAMS_PATH}.')


if __name__ == '__main__':
    from ur5_vision import load_params, CALIBRATION_PARAMS_PATH as PATH
    import os

    if os.path.exists(PATH):
        load_params()

    camera = Camera.from_device(CAMERA_CONFIG['index'], CAMERA_CONFIG['buffer_size'])
    Calibration(camera).run()
    camera.release()
    cv2.destroyAllWindows()
    print('--- Calibration complete ---')
