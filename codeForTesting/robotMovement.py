#Anden del:
import sys
sys.path.append(r"C:\RoboDK\Python") #Makes sure we find the right path to connect robodk and python

import robodk.robolink as rl #Imports the libary so we can 
from robodk.robolink import RUNMODE_RUN_ROBOT
import time

RDK = rl.Robolink()
RDK.setRunMode(RUNMODE_RUN_ROBOT)
robot = RDK.Item('UR5',rl.ITEM_TYPE_ROBOT)

def init(rdk,rob):
    global RDK, robot
    RDK = rdk
    robot = rob

def getTarget(name): #Defines the funktion
    target = RDK.Item(name, rl.ITEM_TYPE_TARGET) #Creates a local variable, that locates a target
    if not target.Valid(): #checks if the target is valid
        raise Exception (f'Target {name}, could not be found') #Closes the program and returns and error message instead of the entire program crashes
    return target #Returns the valid target

def openGripper():
    robot.setDO(7,0)
    robot.setDO(6,1)
    time.sleep(0.5)

def closeGripper():
    robot.setDO(6,0)
    robot.setDO(7,1)
    time.sleep(0.5)

def moveToPlace(position):
    robot.MoveJ(getTarget(position))

def pickOrPlace(above,on,gripper='close'):
    robot.MoveL(getTarget(on))
    if gripper=='close':
        closeGripper()
    else:
        openGripper()
    robot.MoveL(getTarget(above))