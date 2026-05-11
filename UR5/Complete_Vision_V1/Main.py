"""Main program for UR5 EV battery disassembly system.

Initialises the vision system once on startup, then runs a production
loop processing batteries until the operator shuts down.
"""

import cv2
import numpy as np

from ur5_vision import (
    BAR_HEIGHT, BLOB_CONFIG, CAMERA_CONFIG, MORPHOLOGY_CONFIG,
    Camera, Detector, DetectionResult, Homography, load_params,
)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _show_detection(
    camera: Camera,
    detector: Detector,
    result: DetectionResult,
) -> None:
    """Displays the detection result with numbered blobs and an info bar.

    Shows a green circle and module number on each detected blob. A black
    bar below the image lists each module number, colour, and world position.

    Args:
        camera: Camera instance for frame acquisition.
        detector: Detector instance for mask building.
        result: DetectionResult from the last detection pass.
    """
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
        cv2.circle(display, (cx, cy), 20, (0, 255, 0), 2)  # Green circle outline.
        cv2.circle(display, (cx, cy), 4, colour, -1)        # Red or blue centre dot.
        cv2.putText(
            display, str(i + 1),
            (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    bar_height = 30 + len(result.positions) * 22
    bar = np.zeros((bar_height, display.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, 'Module  Colour  World X   World Y', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    for i, ((wx, wy), red) in enumerate(zip(result.positions, result.is_red)):
        colour = (0, 0, 255) if red else (255, 0, 0)
        cv2.putText(
            bar,
            f'  {i + 1}       {"Red " if red else "Blue"}    {wx:6.1f}    {wy:6.1f}',
            (10, 42 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
        )

    cv2.imshow('Detection Result', np.vstack([display, bar]))
    cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Operator prompt
# ---------------------------------------------------------------------------

def _ask_operator(question: str) -> bool:
    """Displays a question and waits for Y or N keypress.

    Args:
        question: Question to display.

    Returns:
        True if Y pressed, False if N pressed.
    """
    print(f'\n{question} (Y/N)')
    prompt = np.zeros((80, 700, 3), dtype=np.uint8)
    cv2.putText(prompt, question, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        prompt, 'Y = confirm   N = cancel',
        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )
    cv2.imshow('UR5 System', prompt)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('y'):
            return True
        if key == ord('n'):
            return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Runs the UR5 battery disassembly production loop."""
    print('--- UR5 Battery Disassembly System ---')

    load_params()

    camera = Camera.from_device(CAMERA_CONFIG['index'], CAMERA_CONFIG['buffer_size'])
    homography = Homography(camera)
    homography.setup()
    detector = Detector(camera, homography)

    if not _ask_operator('Start battery disassembly?'):
        camera.release()
        cv2.destroyAllWindows()
        return

    while True:
        # # Robot: remove lid here.

        result = detector.detect()
        print(f'Detected {len(result.positions)} module(s).')
        print(f'Positions: {result.positions}')
        print(f'Is red:    {result.is_red}')

        _show_detection(camera, detector, result)

        if len(result.positions) > 8:
            if not _ask_operator(f'{len(result.positions)} modules detected. Continue?'):
                if not _ask_operator('Run again?'):
                    break
                continue

        if not _ask_operator('Run again with next battery?'):
            break

    camera.release()
    cv2.destroyAllWindows()
    print('--- System shut down ---')


if __name__ == '__main__':
    main()