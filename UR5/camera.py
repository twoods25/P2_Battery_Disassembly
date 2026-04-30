"""
camera.py
---------
Captures a single snapshot on startup, detects red/blue blocks inside a
crop/zoom window, and returns a structured work order for the robot loop.

Output: a list of 8 dicts, sorted left-to-right / top-to-bottom:
    [
        {'index': 0, 'color': 'red',  'pick': True,  'pixel': (cx, cy)},
        {'index': 1, 'color': 'blue', 'pick': False, 'pixel': (cx, cy)},
        ...
    ]

'pick' is True for red blocks, False for blue — change the rule in should_pick().

--- Workflow for setting up the crop window ---
1. Run:  python camera.py --adjust
2. Use the trackbars to center and zoom the window over the 8 modules.
3. The console prints the final CROP values when you press Q.
4. Paste those values into the CROP dict below and you are done.
"""

import cv2
import numpy as np
import argparse
from util import get_limits

# ─────────────────────────────────────────────
# Configuration  (tune these values as needed)
# ─────────────────────────────────────────────

CAMERA_INDEX   = 1        # which camera to open (0 = built-in, 1 = external)
EXPECTED_COUNT = 8        # how many blocks we expect to find
MIN_AREA       = 500      # minimum contour area to count as a real block (filters noise)

# BGR values for each color we want to detect
COLORS = {
    'red':  [0, 0, 255],
    'blue': [255, 0, 0],
}

# ─────────────────────────────────────────────
# Crop / zoom configuration
# ─────────────────────────────────────────────
# Run `python camera.py --adjust` to tune these values visually.
# Paste the printed output back here once you are happy with the crop window.
#
#   center_x / center_y : where to point the crop (0.0 = left/top, 1.0 = right/bottom)
#   zoom                 : how tight the crop is  (1.0 = full frame, 3.0 = one third, etc.)

CROP = {
    'center_x': 0.5,    # horizontal center of the work area (0.0 – 1.0)
    'center_y': 0.5,    # vertical center of the work area   (0.0 – 1.0)
    'zoom':     1.5,    # zoom level — higher = tighter/smaller crop window
}


# ─────────────────────────────────────────────
# Decision rule  (single place to change logic)
# ─────────────────────────────────────────────

def should_pick(color: str) -> bool:
    """Return True if this block color should be picked. Red = pick, blue = skip."""
    return color == 'red'


# ─────────────────────────────────────────────
# Crop helper
# ─────────────────────────────────────────────

def apply_crop(frame, crop: dict) -> np.ndarray:
    """
    Crop and resize a region of the frame based on center position and zoom.

    How it works:
      - zoom=1.0  → full frame (no crop)
      - zoom=2.0  → crop a window that is half the frame width/height
      - zoom=3.0  → crop a window that is a third of the frame width/height
    The cropped region is then resized back to the original frame dimensions
    so all downstream code sees the same image size.
    """
    h, w = frame.shape[:2]

    # Calculate the width/height of the crop window
    crop_w = int(w / crop['zoom'])
    crop_h = int(h / crop['zoom'])

    # Calculate the top-left corner of the crop window from the center point
    cx = int(crop['center_x'] * w)
    cy = int(crop['center_y'] * h)
    x1 = max(cx - crop_w // 2, 0)
    y1 = max(cy - crop_h // 2, 0)

    # Make sure the window does not go outside the frame boundaries
    x1 = min(x1, w - crop_w)
    y1 = min(y1, h - crop_h)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Crop and scale back to original resolution
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────

def detect_blocks(hsv_frame, color_name, bgr_value) -> list:
    """
    Detect all blocks of a given color in an HSV frame.
    Returns a list of dicts with bbox and center pixel for each block found.
    """
    # Build a binary mask where pixels matching the color are white
    lower, upper = get_limits(color=bgr_value)
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Morphological open removes small noise specks (erode then dilate)
    # Morphological close fills small holes inside detected regions (dilate then erode)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours (outlines) of white regions in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue  # too small — likely noise, skip it

        x, y, w, h = cv2.boundingRect(contour)
        blocks.append({
            'color': color_name,
            'bbox':  (x, y, x + w, y + h),       # (x1, y1, x2, y2)
            'pixel': (x + w // 2, y + h // 2),    # center point (cx, cy)
        })

    return blocks


def build_work_order(frame) -> list:
    """
    Detect all colored blocks in a single frame and return a sorted work order.
    Sorted left-to-right (by X), then top-to-bottom (by Y) within same column.
    Each entry gets a 'pick' flag and a sequential index.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Collect blocks for every configured color
    all_blocks = []
    for color_name, bgr_value in COLORS.items():
        all_blocks.extend(detect_blocks(hsv, color_name, bgr_value))

    # Sort: primary = X (left to right), secondary = Y (top to bottom)
    all_blocks.sort(key=lambda b: (b['pixel'][0], b['pixel'][1]))

    # Assign index and pick decision to each block
    for i, block in enumerate(all_blocks):
        block['index'] = i
        block['pick']  = should_pick(block['color'])

    return all_blocks


# ─────────────────────────────────────────────
# Visualisation  (drawn on the snapshot preview)
# ─────────────────────────────────────────────

def draw_results(frame, work_order) -> None:
    """
    Draw bounding boxes, index numbers, and PICK/SKIP labels on the frame.
    Modifies the frame in place.
    """
    style = {
        'red':  {'border': (0, 0, 255), 'fill': (0, 0, 200)},
        'blue': {'border': (255, 0, 0), 'fill': (200, 0, 0)},
    }

    for block in work_order:
        x1, y1, x2, y2 = block['bbox']
        cx, cy          = block['pixel']
        s               = style[block['color']]

        # Semi-transparent filled rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), s['fill'], -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), s['border'], 2)

        # Index label top-left, PICK/SKIP label center
        cv2.putText(frame, f"#{block['index']}",
                    (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        label       = "PICK" if block['pick'] else "SKIP"
        label_color = (0, 255, 0) if block['pick'] else (0, 165, 255)
        cv2.putText(frame, label,
                    (cx - 18, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    cv2.putText(frame, f"Detected: {len(work_order)}/{EXPECTED_COUNT} blocks",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press any key to continue ...",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# ─────────────────────────────────────────────
# Crop adjustment mode
# ─────────────────────────────────────────────

def adjust_crop():
    """
    Interactive live mode for tuning the crop window.

    Opens the camera feed with three trackbars:
        Center X  — horizontal position of the crop center
        Center Y  — vertical position of the crop center
        Zoom x10  — zoom level (10 = 1.0x, 30 = 3.0x, etc.)

    A green rectangle shows exactly what area will be cropped.
    Press Q to exit — the final values are printed to the console
    so you can paste them into the CROP dict above.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[camera] ERROR: could not open camera index {CAMERA_INDEX}")
        return

    win = "Crop Adjustment  (Q to finish)"
    cv2.namedWindow(win)

    # Trackbars use integers, so we scale:
    #   center: 0–100 maps to 0.0–1.0
    #   zoom:   10–50 maps to 1.0–5.0  (divided by 10)
    cv2.createTrackbar("Center X", win, int(CROP['center_x'] * 100), 100, lambda v: None)
    cv2.createTrackbar("Center Y", win, int(CROP['center_y'] * 100), 100, lambda v: None)
    cv2.createTrackbar("Zoom x10", win, int(CROP['zoom'] * 10),       50,  lambda v: None)

    print("[camera] Adjust the trackbars to frame the work area, then press Q.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Read current trackbar values and convert back to floats
        cx_pct = cv2.getTrackbarPos("Center X", win) / 100
        cy_pct = cv2.getTrackbarPos("Center Y", win) / 100
        zoom   = max(cv2.getTrackbarPos("Zoom x10", win) / 10, 1.0)  # minimum 1.0

        # Calculate the crop rectangle in pixel coordinates for the preview overlay
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        cx_px  = int(cx_pct * w)
        cy_px  = int(cy_pct * h)
        rx1    = max(cx_px - crop_w // 2, 0)
        ry1    = max(cy_px - crop_h // 2, 0)
        rx1    = min(rx1, w - crop_w)
        ry1    = min(ry1, h - crop_h)
        rx2    = rx1 + crop_w
        ry2    = ry1 + crop_h

        # Dim everything outside the crop window so the selection stands out
        dimmed = frame.copy()
        dimmed[:, :]           = (frame * 0.35).astype(np.uint8)  # dim whole frame
        dimmed[ry1:ry2, rx1:rx2] = frame[ry1:ry2, rx1:rx2]        # restore crop area

        # Draw green crop border and current values
        cv2.rectangle(dimmed, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.putText(dimmed, f"zoom={zoom:.1f}x  center=({cx_pct:.2f}, {cy_pct:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(dimmed, "Press Q when happy with the crop",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(win, dimmed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Print the final values ready to paste into CROP above
            print("\n[camera] Paste these values into the CROP dict in camera.py:")
            print( "CROP = {")
            print(f"    'center_x': {cx_pct:.2f},")
            print(f"    'center_y': {cy_pct:.2f},")
            print(f"    'zoom':     {zoom:.1f},")
            print( "}")
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def capture_work_order() -> list:
    """
    Open the camera, grab ONE snapshot, apply the crop window, detect blocks,
    show the result, and return the work order list.

    Call this once at program startup before the robot loop begins.
    Returns an empty list if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[camera] ERROR: could not open camera index {CAMERA_INDEX}")
        return []

    ret, frame = cap.read()   # single snapshot — no continuous loop needed
    cap.release()             # release the camera immediately to free resources

    if not ret:
        print("[camera] ERROR: failed to read frame from camera")
        return []

    # Apply the crop/zoom window — only the work area is evaluated from here on
    frame = apply_crop(frame, CROP)

    # Build the work order from the cropped snapshot
    work_order = build_work_order(frame)

    if len(work_order) != EXPECTED_COUNT:
        print(f"[camera] WARNING: expected {EXPECTED_COUNT} blocks, "
              f"found {len(work_order)}")

    # Show the annotated result until a key is pressed
    draw_results(frame, work_order)
    cv2.imshow("Snapshot — Block Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print work order summary to console
    print("\n[camera] Work order:")
    for block in work_order:
        action = "PICK" if block['pick'] else "SKIP"
        print(f"  #{block['index']}  {block['color'].upper():4s}  →  {action}  "
              f"pixel=({block['pixel'][0]}, {block['pixel'][1]})")

    return work_order


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Run with --adjust to enter the interactive crop tuning mode
    # Run with no arguments for a normal detection snapshot
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjust", action="store_true",
                        help="Open interactive crop adjustment window")
    args = parser.parse_args()

    if args.adjust:
        adjust_crop()
    else:
        work_order = capture_work_order()
        pick_flags = [b['pick'] for b in work_order]
        print(f"\nRobot pick array: {pick_flags}")