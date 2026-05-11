"""
positions.py
------------
Function with predefined relative module positions that defines
pick and place positions for the the pick and place task based
on knowing the corner of the battery pack.

The battery has 8 modules arranged in a 2x4 grid.
Measure origin of mockup battery (front-left corner) relative to robot base frame,
and insert in this function at top.

All 8 module positions are calculated automatically from the grid layout.

Coordinate system:
  - Origin = front-left corner of the battery (measured in robot base frame, mm)
  - X axis = along the long side of the battery (column direction)
  - Y axis = across the short side of the battery (row direction)
  - Z axis = up/down

    Col 0     Col 1
    ┌─────┐   ┌─────┐
    │  0  │   │  1  │   ← Row 0
    ├─────┤   ├─────┤
    │  2  │   │  3  │   ← Row 1
    ├─────┤   ├─────┤
    │  4  │   │  5  │   ← Row 2
    ├─────┤   ├─────┤
    │  6  │   │  7  │   ← Row 3
    └─────┘   └─────┘
"""

# Origin: front-left corner of the battery in robot base coordinates:
battery_origin = (400.0, 200.0, 50.0)   # (x, y, z) (robot base frame, mm) USER INPUT!

# Grid spacing between module centers (mm)
module_spacing_x = 60.0    # distance between columns
module_spacing_y = 80.0    # distance between rows

# Grid layout: 2 columns, 4 rows
grid_col = 2
grid_row = 4

# z heights for pick motion (relative to z of battery origin)
pick_height   =  0.0    # z height when picking (at module surface)
approach_height = 50.0  # z height for approach/retreat above the module

# Drop-off position — where all picked modules are dropped off (robot base frame, mm)
place_position = (600.0, 300.0, 100.0)  # (x, y, z) — USER INPUT!
place_approach = (600.0, 300.0, 150.0)  # approach point above the place position


# ─────────────────────────────────────────────
# Position calculator
# ─────────────────────────────────────────────

def get_module_positions() -> list:
    """
    Calculate the pick position for each of the 8 modules based on the
    battery origin and grid spacing.

    Returns a list of 8 dicts, one per module, indexed 0–7:
        [
            {'index': 0, 'pick': (x, y, z), 'approach': (x, y, z)},
            ...
        ]

    Index order matches the camera work order (left-to-right, top-to-bottom).
    """
    ox, oy, oz = battery_origin
    positions  = []

    for row in range(grid_row):
        for col in range(grid_col):

            # Calculate module center by offset from origin
            x = ox + col * module_spacing_x
            y = oy + row * module_spacing_y
            z = oz + pick_height

            # Approach point above pick point
            z_approach = oz + approach_height

            index = row * grid_col + col   # numbering 0–7, matching camera index order

            positions.append({
                'index':    index,
                'pick':     (x, y, z),           # position to pick from
                'approach': (x, y, z_approach),  # position to approach/retreat from
            })

    return positions


# ─────────────────────────────────────────────
# Run standalone to verify positions
# ─────────────────────────────────────────────

if __name__ == "__main__":
    positions = get_module_positions()
    print(f"Battery origin: {battery_origin}")
    print(f"Place position: {place_position}\n")
    print("Module positions:")
    for p in positions:
        print(f"  #{p['index']}  pick={p['pick']}  approach={p['approach']}")