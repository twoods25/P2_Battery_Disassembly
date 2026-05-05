"""
camera.py
---------
Snapshot-based color blob detector for pick-and-place work order generation.

Internally the blob detector represents each detected region as a circle
with a center point (kp.pt) and diameter (kp.size). We draw rectangles
over these for clarity, but the detection itself is circle-based.

Setup workflow (run once):
    python camera.py --focus    →  tune focus, paste value into FOCUS below
    python camera.py --adjust   →  tune trim, paste values into CROP below
    python camera.py            →  verify detection, tune BLOB params if needed
"""

import cv2
import numpy as np
import argparse
from util import get_limits

# ═══════════════════════════════════════════════════════════════
# PARAMETERS  —  only edit values in this section
# ═══════════════════════════════════════════════════════════════

# Camera
CAMERA_INDEX = 1
RESOLUTION   = (1920, 1080)
FOCUS        = 0          # manual focus value (run --focus to tune)

# Detection
EXPECTED_COUNT = 8
COLORS = {
    'red':  [0, 0, 255],
    'blue': [255, 0, 0],
}

# Trim  —  fraction to cut from each side (0.0 = none, 0.5 = half)
# Run --adjust to tune these visually
CROP = {
    'trim_top':    0.04,
    'trim_bottom': 0.15,
    'trim_left':   0.16,
    'trim_right':  0.52,
}

# Blob detector parameters
# The detector internally fits a circle to each blob region.
# kp.pt   = center of the circle (x, y)
# kp.size = diameter of the circle
#
# min/max_area     — controls the valid size range for a module blob
# min_circularity  — how close to a perfect circle (1.0). Keep low for rectangles
# min_convexity    — how convex the shape must be. Keep low for notched modules
# min_inertia      — how elongated the shape can be. 0.01 allows any elongation
BLOB = {
    'min_threshold':   10,
    'max_threshold':   200,
    'min_area':        500,    # lower catches smaller/partial blobs
    'max_area':        25000,  # reduced to prevent two adjacent modules merging
    'min_circularity': 0.05,   # very low — modules are rectangular not circular
    'min_convexity':   0.1,    # low — side notches reduce convexity
    'min_inertia':     0.01,   # low — allows elongated shapes
}

# Morphological kernel size
# Smaller = less gap-filling between adjacent modules (prevents merging)
# Larger  = better at filling holes/notches inside a single module
MORPH_KERNEL = 11

# ═══════════════════════════════════════════════════════════════
# END OF PARAMETERS
# ═══════════════════════════════════════════════════════════════


def should_pick(color: str) -> bool:
    """Pick rule — red = pick, blue = skip. Change here if logic changes."""
    return color == 'red'


def configure_camera(cap) -> None:
    """Lock resolution and disable autofocus."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, FOCUS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[camera] {w:.0f}x{h:.0f}  focus={FOCUS}")


def apply_trim(frame) -> np.ndarray:
    """Slice the frame to the work area — no rescaling, no quality loss."""
    h, w = frame.shape[:2]
    y1 = int(h * CROP['trim_top'])
    y2 = int(h * (1.0 - CROP['trim_bottom']))
    x1 = int(w * CROP['trim_left'])
    x2 = int(w * (1.0 - CROP['trim_right']))
    return frame[y1:y2, x1:x2]


def make_blob_detector() -> cv2.SimpleBlobDetector:
    """
    Build the blob detector from BLOB parameters.
    The detector fits a circle to each blob and returns center + diameter.
    """
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold        = BLOB['min_threshold']
    p.maxThreshold        = BLOB['max_threshold']
    p.filterByArea        = True
    p.minArea             = BLOB['min_area']
    p.maxArea             = BLOB['max_area']
    p.filterByCircularity = True
    p.minCircularity      = BLOB['min_circularity']
    p.filterByConvexity   = True
    p.minConvexity        = BLOB['min_convexity']
    p.filterByInertia     = True
    p.minInertiaRatio     = BLOB['min_inertia']
    p.filterByColor       = False
    return cv2.SimpleBlobDetector_create(p)


# Single shared detector instance
DETECTOR = make_blob_detector()


def detect_blobs(frame, color_name, bgr_value) -> list:
    """
    Full detection pipeline for one color:
      blur → HSV mask → open → close → invert → blob detect

    The blob detector returns keypoints where:
      kp.pt   = (cx, cy) center of the fitted circle
      kp.size = diameter of the fitted circle

    We convert the circle into a square bbox for display.
    """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower, upper = get_limits(color=bgr_value)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # remove noise specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill holes/notches

    # Invert — blob detector needs dark blobs on white background
    keypoints = DETECTOR.detect(cv2.bitwise_not(mask))

    # Convert each keypoint circle into a block dict
    return [
        {
            'color': color_name,
            'pixel': (int(kp.pt[0]), int(kp.pt[1])),          # circle center
            'bbox':  (int(kp.pt[0] - kp.size / 2),            # square bbox
                      int(kp.pt[1] - kp.size / 2),            # derived from
                      int(kp.pt[0] + kp.size / 2),            # circle radius
                      int(kp.pt[1] + kp.size / 2)),
        }
        for kp in keypoints
    ]


def build_work_order(frame) -> list:
    """
    Detect all blocks, sort left-to-right then top-to-bottom,
    assign index and pick flag to each.
    """
    all_blocks = []
    for color_name, bgr_value in COLORS.items():
        all_blocks.extend(detect_blobs(frame, color_name, bgr_value))

    # Sort: left to right (X), then top to bottom (Y)
    all_blocks.sort(key=lambda b: (b['pixel'][0], b['pixel'][1]))

    for i, block in enumerate(all_blocks):
        block['index'] = i
        block['pick']  = should_pick(block['color'])

    return all_blocks


def draw_results(frame, work_order) -> None:
    """Draw boxes, center dots, and labels on the frame in place."""
    style = {
        'red':  {'border': (0, 0, 255), 'fill': (0, 0, 180)},
        'blue': {'border': (255, 0, 0), 'fill': (180, 0, 0)},
    }

    for b in work_order:
        x1, y1, x2, y2 = b['bbox']
        cx, cy          = b['pixel']
        s               = style[b['color']]

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), s['fill'], -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), s['border'], 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)  # blob center dot

        cv2.putText(frame, f"#{b['index']}",
                    (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        label       = "PICK" if b['pick'] else "SKIP"
        label_color = (0, 255, 0) if b['pick'] else (0, 165, 255)
        cv2.putText(frame, label,
                    (cx - 18, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    cv2.putText(frame, f"Detected: {len(work_order)}/{EXPECTED_COUNT}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press any key to continue ...",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# ─────────────────────────────────────────────
# Adjustment modes
# ─────────────────────────────────────────────

def adjust_trim():
    """
    Live trim adjustment with 4 trackbars (top/bottom/left/right).
    Press Q — prints final CROP values to paste into parameters above.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    configure_camera(cap)

    win = "Trim Adjustment  (Q to finish)"
    cv2.namedWindow(win)

    for side, key in [("Top", 'trim_top'), ("Bottom", 'trim_bottom'),
                      ("Left", 'trim_left'), ("Right", 'trim_right')]:
        cv2.createTrackbar(f"Trim {side}", win, int(CROP[key] * 100), 100, lambda v: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        t = cv2.getTrackbarPos("Trim Top",    win) / 100
        b = cv2.getTrackbarPos("Trim Bottom", win) / 100
        l = cv2.getTrackbarPos("Trim Left",   win) / 100
        r = cv2.getTrackbarPos("Trim Right",  win) / 100

        y1, y2 = int(h * t), int(h * (1 - b))
        x1, x2 = int(w * l), int(w * (1 - r))

        dimmed = (frame * 0.35).astype(np.uint8)
        dimmed[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        cv2.rectangle(dimmed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(dimmed, f"T={t:.2f} B={b:.2f} L={l:.2f} R={r:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(dimmed, "Press Q when done",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow(win, dimmed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[camera] Paste into CROP in camera.py:")
            print(f"CROP = {{'trim_top': {t:.2f}, 'trim_bottom': {b:.2f}, "
                  f"'trim_left': {l:.2f}, 'trim_right': {r:.2f}}}")
            break

    cap.release()
    cv2.destroyAllWindows()


def adjust_focus():
    """
    Live focus adjustment with a single trackbar.
    Press Q — prints final focus value to paste into FOCUS above.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    win = "Focus Adjustment  (Q to finish)"
    cv2.namedWindow(win)
    cv2.createTrackbar("Focus", win, FOCUS, 255, lambda v: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        focus = cv2.getTrackbarPos("Focus", win)
        cap.set(cv2.CAP_PROP_FOCUS, focus)
        cv2.putText(frame, f"Focus: {focus}  (Q to save)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n[camera] Paste into parameters: FOCUS = {focus}")
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def capture_work_order() -> list:
    """
    Grab one snapshot, trim, detect, display, return work order.
    Call once at program startup before the robot loop.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[camera] ERROR: could not open camera {CAMERA_INDEX}")
        return []

    configure_camera(cap)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[camera] ERROR: failed to capture frame")
        return []

    frame      = apply_trim(frame)
    work_order = build_work_order(frame)

    if len(work_order) != EXPECTED_COUNT:
        print(f"[camera] WARNING: expected {EXPECTED_COUNT}, found {len(work_order)}")

    draw_results(frame, work_order)
    cv2.imshow("Block Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[camera] Work order:")
    for b in work_order:
        print(f"  #{b['index']}  {b['color'].upper():4s}  "
              f"{'PICK' if b['pick'] else 'SKIP'}  pixel={b['pixel']}")

    return work_order


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjust", action="store_true", help="Tune trim window")
    parser.add_argument("--focus",  action="store_true", help="Tune focus")
    args = parser.parse_args()

    if args.adjust:
        adjust_trim()
    elif args.focus:
        adjust_focus()
    else:
        work_order = capture_work_order()
        print(f"\nPick array: {[b['pick'] for b in work_order]}")