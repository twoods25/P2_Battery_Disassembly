#This is the main program
#Controls Vision, Robot, Positions
#
#Vision Guided Pick-and-Place System

from camera import capture_work_order
from positions import get_module_positions
from robot import run_pick_and_place

def main():
    #Step 1 - Vision:
    #Snapshot Camera -> Detect Module Quantity (8) -> Return Work Order'
    #Sorts Red from Blue (Pick vs Skip)
    work_order = capture_work_order()

    if not work_order:
        print("[main] No modules detected - aborting operation.")
        return

    #Step 2 - Positions:
    #Userinput for Battery Origin Point (Coordinates of Corner)
    #Module Positions calculated automatically (predefined relative to corner)
    positions = get_module_positions()

    #Step 3 - Robot:
    #Connects to RoboDK and executes pick/place order for "PICK" modules.
    run_pick_and_place(work_order, positions)

if __name__ == "__main__":
    main()
