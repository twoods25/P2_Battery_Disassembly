import sys
sys.path.append(r"C:\RoboDK\Python")
import robodk 

import robodk.robolink as rl
from robodk.robolink import RUNMODE_RUN_ROBOT
import robotMovement as rm
from robotMovement import * 

RDK = rl.Robolink()
RDK.setRunMode(RUNMODE_RUN_ROBOT)
robot = RDK.Item('UR5', rl.ITEM_TYPE_ROBOT)

rm.init(RDK,robot)

#Main code

rm.moveToPlace("Home Position")

rm.moveToPlace("Batteri låg over")

#rm.pickOrPlace("Batteri låg over", "Batteri låg på",gripper="close")

rm.moveToPlace("Låg dump over")

#rm.pickOrPlace("Låg dump over", "Låg dump plads",gripper="open")

#Modules =  [True, False, False, True, False, True, True, False] #The values should be based off the camera so these are temporary

Modules = [True,True,True,True,True,True,True,True] #For testing purposes, to ensure the robot can move to all approach positions

slotTæller = 1

dumpTæller = 1

for Position in Modules:
    if Position:
        rm.moveToPlace(f"Batteri modul {slotTæller} over")
        #rm.pickOrPlace(f"Batteri modul {slotTæller} over", f"Batteri modul {slotTæller} på", gripper="close")
        rm.moveToPlace(f"Modul dump {dumpTæller} over")
        #rm.pickOrPlace(f"Modul dump {dumpTæller} over",f"Modul dump {dumpTæller} på",gripper="open")
        slotTæller += 1
        dumpTæller += 1
    else:
        slotTæller += 1

rm.moveToPlace("Home Position")
print("All bad modules have been removed")