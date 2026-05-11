"""UR5 UI module for EV battery disassembly robot.

All display, prompts, and user interaction. No pipeline logic.
Can be removed entirely without affecting ur5_vision.py.

Depends on: ur5_vision.py
"""

import os

import cv2
import numpy as np

from ur5_vision import (
    CALIBRATION_TARGETS_WORLD,
    CORNER_LABELS,
    HOMOGRAPHY_SAVE_PATH,
    Camera,
    DetectionResult,
    Homography,
)

_BAR_HEIGHT = 60   # Height of instruction bar above setup images in pixels.
_PROMPT_WIN = 'UR5 System'


# ---------------------------------------------------------------------------
# Operator prompt
# ---------------------------------------------------------------------------

def ask_operator(question: str) -> bool:
    """Displays a question window and waits for Y or N keypress.

    Args:
        question: Question text to display.

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
    cv2.imshow(_PROMPT_WIN, prompt)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('y'):
            return True
        if key == ord('n'):
            return False


# ---------------------------------------------------------------------------
# Detection display
# ---------------------------------------------------------------------------

def show_detection(camera: Camera, result: DetectionResult) -> None:
    """Displays detection result with numbered blobs and an info bar.

    Draws a green circle and module number on each blob in the image.
    A black bar below the image lists module number, colour, and world position.
    Image and info bar are built as separate arrays and only stacked at display time.

    Args:
        camera: Camera instance for a fresh display frame.
        result: DetectionResult from the last detection pass.
    """
    frame = camera.get_cropped_frame()  # Fresh frame for display only.
    display = frame.copy()

    for i, ((px, py), (wx, wy), red) in enumerate(
        zip(result.pixel_centroids, result.positions, result.is_red)
    ):
        cx, cy = int(px), int(py)
        colour = (0, 0, 255) if red else (255, 0, 0)
        cv2.circle(display, (cx, cy), 20, (0, 255, 0), 2)   # Green circle outline.
        cv2.circle(display, (cx, cy), 4, colour, -1)          # Red or blue centre dot.
        cv2.putText(
            display, str(i + 1),
            (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    bar_height = 30 + len(result.positions) * 22
    bar = np.zeros((bar_height, display.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar, 'Module  Colour  World X   World Y',
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )
    for i, ((wx, wy), red) in enumerate(zip(result.positions, result.is_red)):
        colour = (0, 0, 255) if red else (255, 0, 0)
        cv2.putText(
            bar,
            f'  {i + 1}       {"Red " if red else "Blue"}    {wx:6.1f}    {wy:6.1f}',
            (10, 42 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
        )

    cv2.imshow('Detection Result', np.vstack([display, bar]))
    cv2.waitKey(1)


def save_detection_image(camera: Camera, result: DetectionResult) -> None:
    """Saves the detection result image to disk.

    Saves as result.jpg. If the file already exists, saves as
    result(1).jpg, result(2).jpg and so on.

    Args:
        camera: Camera instance for a fresh display frame.
        result: DetectionResult from the last detection pass.
    """
    frame = camera.get_cropped_frame()
    display = frame.copy()
    for i, ((px, py), (wx, wy), red) in enumerate(
        zip(result.pixel_centroids, result.positions, result.is_red)
    ):
        cx, cy = int(px), int(py)
        colour = (0, 0, 255) if red else (255, 0, 0)
        cv2.circle(display, (cx, cy), 20, (0, 255, 0), 2)
        cv2.circle(display, (cx, cy), 4, colour, -1)
        cv2.putText(
            display, str(i + 1),
            (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
    path = 'result.jpg'
    counter = 1
    while os.path.exists(path):
        path = f'result({counter}).jpg'
        counter += 1
    cv2.imwrite(path, display)
    print(f'Detection image saved to {path}.')


# ---------------------------------------------------------------------------
# Homography setup UI
# ---------------------------------------------------------------------------

def run_homography_setup(camera: Camera, homography: Homography) -> None:
    """Interactive click interface for setting up the homography matrix.

    Displays the cropped frame with an instruction bar. The user clicks
    4 calibration targets in order: Top-Left, Top-Right, Bottom-Right,
    Bottom-Left. Offers to load a previously saved matrix on startup.

    Calls homography.compute() with the collected pixel points, which
    saves the matrix to file.

    Args:
        camera: Camera instance for frame acquisition.
        homography: Homography instance to compute and store the matrix.
    """
    if homography.is_ready:
        frame = camera.get_cropped_frame()
        bar = np.zeros((_BAR_HEIGHT, frame.shape[1], 3), dtype=np.uint8)
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
            elif key == ord('q'):
                cv2.destroyAllWindows()
                raise RuntimeError('Homography setup cancelled by user.')

    base_frame = camera.get_cropped_frame()
    clicks = []
    mouse_pos = [None]

    def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos[0] = (x, y - _BAR_HEIGHT)  # Adjust for instruction bar offset.
        elif event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y - _BAR_HEIGHT))  # Adjust for instruction bar offset.
            print(
                f'Click {len(clicks)}: {CORNER_LABELS[len(clicks) - 1]}'
                f' at ({x}, {y - _BAR_HEIGHT}).'
            )

    cv2.namedWindow('Homography Setup')
    cv2.setMouseCallback('Homography Setup', on_mouse)

    while True:
        display = _draw_click_state(base_frame, clicks, mouse_pos[0])
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
    homography.compute(pixel_points)
    print('Homography saved.')


def _draw_click_state(
    base_frame: np.ndarray,
    clicks: list[tuple[int, int]],
    mouse_pos: tuple[int, int] | None,
) -> np.ndarray:
    """Draws click state and rubber band line on the frame with an instruction bar.

    Args:
        base_frame: Clean unmodified cropped frame.
        clicks: Clicked pixel coordinates in image space.
        mouse_pos: Current mouse position in image space for rubber band line.

    Returns:
        Frame with dots, lines, and instruction bar stacked above.
    """
    display = base_frame.copy()
    for i, point in enumerate(clicks):
        cv2.circle(display, point, 6, (0, 255, 0), -1)
        cv2.putText(
            display, CORNER_LABELS[i],
            (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        if i > 0:
            _draw_dotted_line(display, clicks[i - 1], point, (0, 255, 0))
    if len(clicks) == 4:
        _draw_dotted_line(display, clicks[3], clicks[0], (0, 165, 255))
    if clicks and mouse_pos is not None and len(clicks) < 4:
        _draw_dotted_line(display, clicks[-1], mouse_pos, (0, 200, 255))

    if len(clicks) < 4:
        line1 = f'Click {CORNER_LABELS[len(clicks)]}  ({len(clicks)}/4 selected)'
        line2 = 'U = undo last click   Q = quit'
    else:
        line1 = 'All 4 points selected.'
        line2 = 'Press any key to confirm   U = undo last click'

    bar = np.zeros((_BAR_HEIGHT, display.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return np.vstack([bar, display])


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
