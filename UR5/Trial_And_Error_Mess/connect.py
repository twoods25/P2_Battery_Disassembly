"""
robodk_test.py
--------------
Minimal RoboDK connectivity test for a UR5 (or any 6-axis robot).

Works identically for:
  - RoboDK simulation  (just open RoboDK with a robot loaded)
  - Live UR5 via RoboDK driver  (enable "Run on robot" in RoboDK)

Requirements:
    pip install robodk

Usage:
    python robodk_test.py
"""

from robodk.robolink import Robolink, ITEM_TYPE_ROBOT, RUNMODE_RUN_ROBOT  # RoboDK API
from robodk.robomath import transl, rotz, Mat                             # Matrix helpers
import time


# ─────────────────────────────────────────────
# 1. Connect to RoboDK
# ─────────────────────────────────────────────
def connect() -> tuple:
    """
    Open a connection to RoboDK (must already be running).
    Returns (RDK, robot) or raises if nothing is found.
    """
    print("[1] Connecting to RoboDK ...")
    RDK = Robolink()                        # default: localhost:20500

    # ─────────────────────────────────────────────────────────────────
    # COMMENT OUT the line below to run in SIMULATION
    # UNCOMMENT the line below to run on the REAL UR5
    # (also make sure you connected the robot in RoboDK first:
    #  right-click robot → Connect to robot → enter UR5 IP)
    # ─────────────────────────────────────────────────────────────────
    RDK.setRunMode(RUNMODE_RUN_ROBOT)       # <-- COMMENT THIS OUT FOR SIMULATION

    # Grab the first available robot in the station
    robot = RDK.ItemUserPick("Select a robot", ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError("No robot selected or found in RoboDK station.")

    print(f"    Connected  →  robot: '{robot.Name()}'")
    return RDK, robot


# ─────────────────────────────────────────────
# 2. Safety / configuration helpers
# ─────────────────────────────────────────────
def configure_robot(robot, speed_mms: float = 100, accel_mms: float = 200):
    """Set linear speed and acceleration (mm/s, mm/s²)."""
    robot.setSpeed(speed_mms, accel_mms)
    print(f"[2] Speed set to {speed_mms} mm/s, accel {accel_mms} mm/s²")


# ─────────────────────────────────────────────
# 3. Motion commands
# ─────────────────────────────────────────────
def run_test_sequence(robot):
    """
    Execute a small pick-and-place style move sequence.
    All targets are defined relative to the robot's home pose so
    this runs safely in simulation without a real station setup.
    """
    print("[3] Reading current (home) pose ...")
    home_pose = robot.Pose()               # 4×4 homogeneous matrix
    print(f"    Home TCP position (x,y,z mm): "
          f"{home_pose[0,3]:.1f}, {home_pose[1,3]:.1f}, {home_pose[2,3]:.1f}")

    # --- Move 1: approach (100 mm above home) ---
    print("[4] Move 1 – approach (100 mm up) ...")
    approach = home_pose * transl(0, 0, 100)   # +Z = up in tool frame
    robot.MoveL(approach)
    time.sleep(0.5)

    # --- Move 2: move sideways 150 mm in X ---
    print("[5] Move 2 – shift 150 mm in X ...")
    side = approach * transl(150, 0, 0)
    robot.MoveL(side)
    time.sleep(0.5)

    # --- Move 3: descend back to home height ---
    print("[6] Move 3 – descend ...")
    down = side * transl(0, 0, -100)
    robot.MoveL(down)
    time.sleep(0.5)

    # --- Move 4: joint move back to home ---
    print("[7] Move 4 – joint move back to home ...")
    robot.MoveJ(home_pose)
    time.sleep(0.5)

    print("[✓] Sequence complete – robot back at home.")


# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  RoboDK connectivity test")
    print("=" * 50)

    try:
        RDK, robot = connect()
        configure_robot(robot, speed_mms=80, accel_mms=150)
        run_test_sequence(robot)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nTroubleshooting checklist:")
        print("  • Is RoboDK open and running?")
        print("  • Is a robot loaded in the station?")
        print("  • For live UR5: is 'Run on robot' enabled in RoboDK?")
        print("  • pip install robodk  (if import failed)")
        return

    print("\nNext steps:")
    print("  • Enable 'Run on robot' in RoboDK → same script drives the real UR5")
    print("  • Add your camera/sensor logic between Move 3 and Move 4")
    print("  • Replace relative transl() offsets with real pick-point poses")


if __name__ == "__main__":
    main()