import cv2
import numpy as np
from util import get_limits

# --- Configuration ---
RED_BGR  = [0, 0, 255]
BLUE_BGR = [255, 0, 0]

COLS = 2
ROWS = 4
MIN_BLOCK_AREA = 500  # tune this to filter out noise

def get_mask(hsv_frame, bgr_color):
    lower, upper = get_limits(color=bgr_color)
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_blocks(mask):
    """Find individual block bounding boxes from a mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_BLOCK_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        blocks.append((x, y, w, h))
    return blocks

def build_grid(red_blocks, blue_blocks):
    """
    Combine all detected blocks, sort them into grid order
    (column first, top to bottom, left to right),
    and return a pick list.
    """
    all_blocks = []

    for (x, y, w, h) in red_blocks:
        all_blocks.append({'bbox': (x, y, x+w, y+h), 'color': 'red', 'cx': x + w//2, 'cy': y + h//2})

    for (x, y, w, h) in blue_blocks:
        all_blocks.append({'bbox': (x, y, x+w, y+h), 'color': 'blue', 'cx': x + w//2, 'cy': y + h//2})

    if len(all_blocks) == 0:
        return []

    # Sort by column first (cx), then by row (cy) within each column
    all_blocks.sort(key=lambda b: (b['cx'], b['cy']))

    # Assign position index
    for i, block in enumerate(all_blocks):
        block['position_index'] = i
        block['pick'] = (block['color'] == 'red')

    return all_blocks

def draw_overlay(frame, pick_list):
    for entry in pick_list:
        x1, y1, x2, y2 = entry['bbox']
        color = entry['color']
        pick  = entry['pick']
        i     = entry['position_index']
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Transparent fill
        overlay = frame.copy()
        fill = (0, 0, 200) if color == 'red' else (200, 0, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Border
        border_color = (0, 0, 255) if color == 'red' else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

        # Position index
        cv2.putText(frame, f"#{i}", (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # PICK / SKIP
        label       = "PICK" if pick else "SKIP"
        label_color = (0, 255, 0) if pick else (0, 165, 255)
        cv2.putText(frame, label, (cx - 18, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    return frame

def draw_sidebar(frame, pick_list):
    sidebar_x = frame.shape[1] - 160
    cv2.rectangle(frame, (sidebar_x, 0), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
    cv2.putText(frame, "PICK LIST", (sidebar_x + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for entry in pick_list:
        i     = entry['position_index']
        color = entry['color']
        pick  = entry['pick']
        y_pos = 50 + i * 28

        dot_color    = (0, 0, 255) if color == 'red' else (255, 0, 0)
        action       = "PICK" if pick else "SKIP"
        action_color = (0, 255, 0) if pick else (0, 165, 255)

        cv2.circle(frame, (sidebar_x + 12, y_pos), 7, dot_color, -1)
        cv2.putText(frame, f"{i}: {action}", (sidebar_x + 25, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, action_color, 1)

    return frame


# --- Main Loop ---
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red  = get_mask(hsv, RED_BGR)
    mask_blue = get_mask(hsv, BLUE_BGR)

    red_blocks  = find_blocks(mask_red)
    blue_blocks = find_blocks(mask_blue)

    pick_list = build_grid(red_blocks, blue_blocks)

    frame = draw_overlay(frame, pick_list)
    frame = draw_sidebar(frame, pick_list)

    # Show block count for debugging
    cv2.putText(frame, f"Detected: {len(pick_list)}/8 blocks", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Block Scanner', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("\n--- SNAPSHOT ---")
        for entry in pick_list:
            print(f"  Position {entry['position_index']}: "
                  f"{entry['color'].upper()} → {'PICK' if entry['pick'] else 'SKIP'}")
        final = [entry['pick'] for entry in pick_list]
        print(f"\nRobot pick array: {final}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()