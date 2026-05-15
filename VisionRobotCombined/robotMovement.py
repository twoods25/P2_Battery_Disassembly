import sys
sys.path.append(r"C:\RoboDK\Python")

import robodk.robolink as rl
import robodk.robomath as rmath
import time


WORLD_FRAME_NAME = "World frame"

PICKUP_ORIENTATION_TARGET = "module_position_FIRST"
PICKUP_ORIENTATION_TARGET_LEFT = "module_position_SECOND"

MODULE_APPROACH_HEIGHT = 80
LEFT_MODULE_Y_LIMIT = 160


def init(rdk, rob):
    global RDK, robot
    RDK = rdk
    robot = rob
    #robot.setSpeed(400, 240, 2000, 160)  # linear speed, joint speed, linear accel, joint accel


def getTarget(name):
    target = RDK.Item(name, rl.ITEM_TYPE_TARGET)

    if not target.Valid():
        raise Exception(f"Target '{name}' could not be found")

    return target


def getFrame(name):
    frame = RDK.Item(name, rl.ITEM_TYPE_FRAME)

    if not frame.Valid():
        raise Exception(f"Frame '{name}' could not be found")

    return frame


def useTargetFrame(target):
    """
    Bruges til normale RoboDK targets:
    Home Position, låg targets og dump targets.
    """
    parent = target.Parent()

    if not parent.Valid():
        raise Exception(f"Target '{target.Name()}' has no valid parent frame")

    robot.setPoseFrame(parent)


def useWorldFrame():
    """
    Bruges kun til frie x, y, z modulkoordinater.
    """
    world_frame = getFrame(WORLD_FRAME_NAME)
    robot.setPoseFrame(world_frame)


def getPickupOrientationTarget(x, y):
    """
    Vælger hvilken orientation target robotten skal bruge.

    Hvis modulet ligger langt mod venstre, bruges:
    'Modul pickup orientation left'

    Ellers bruges:
    'Modul pickup orientation'
    """

    if y < LEFT_MODULE_Y_LIMIT:
        return getTarget(PICKUP_ORIENTATION_TARGET_LEFT)

    return getTarget(PICKUP_ORIENTATION_TARGET)


def makeModulePose(x, y, z):
    """
    Laver en pose til modulopsamling:
    - x, y, z kommer fra programmet
    - orienteringen kommer fra det valgte orientation target
    """

    orientation_target = getPickupOrientationTarget(x, y)

    ref_pose = orientation_target.Pose()
    xyzrpw = rmath.pose_2_xyzrpw(ref_pose)

    xyzrpw[0] = x
    xyzrpw[1] = y
    xyzrpw[2] = z

    return rmath.xyzrpw_2_pose(xyzrpw)


def makeModuleTarget(name, x, y, z):
    """
    Laver/flytter et midlertidigt target i World frame.

    target.setJoints(...) hjælper RoboDK med at vælge
    samme UR5-konfiguration som orientation-targetet.
    """

    world_frame = getFrame(WORLD_FRAME_NAME)
    orientation_target = getPickupOrientationTarget(x, y)

    target = RDK.Item(name, rl.ITEM_TYPE_TARGET)

    if not target.Valid():
        target = RDK.AddTarget(name, world_frame, robot)

    target.setParent(world_frame)
    target.setAsCartesianTarget()
    target.setPose(makeModulePose(x, y, z))

    # Hjælper med at undgå wrist flip / forkert UR5-konfiguration
    target.setJoints(orientation_target.Joints())

    return target


def openGripper():
    robot.setDO(7, 0)
    robot.setDO(6, 1)
    time.sleep(0.05)


def closeGripper():
    robot.setDO(6, 0)
    robot.setDO(7, 1)
    time.sleep(0.05)


def moveToPlace(position=None, x=None, y=None, z=None, useTarget=True):
    if useTarget:
        target = getTarget(position)

        # Faste RoboDK-targets bruger altid deres egen frame
        useTargetFrame(target)

        robot.MoveJ(target)

    else:
        # Modulkoordinater bruger World frame
        useWorldFrame()

        # Gå til sikker højde over modulet
        above_target = makeModuleTarget(
            "Temp module move above",
            x,
            y,
            z + MODULE_APPROACH_HEIGHT
        )

        robot.MoveJ(above_target)


def pickOrPlace(
    above=None,
    on=None,
    x=None,
    y=None,
    z=None,
    gripper="close",
    useTarget=True
):
    if useTarget:
        on_target = getTarget(on)
        above_target = getTarget(above)

        # Faste targets bruger targetets egen frame
        useTargetFrame(on_target)
        robot.MoveL(on_target)

        if gripper == "close":
            closeGripper()
        else:
            openGripper()

        useTargetFrame(above_target)
        robot.MoveL(above_target)

    else:
        # Modulkoordinater bruger World frame
        useWorldFrame()

        above_target = makeModuleTarget(
            "Temp module above",
            x,
            y,
            z + MODULE_APPROACH_HEIGHT
        )

        on_target = makeModuleTarget(
            "Temp module on",
            x,
            y,
            z
        )

        robot.MoveL(on_target)

        if gripper == "close":
            closeGripper()
        else:
            openGripper()

        robot.MoveL(above_target)

def pickOrPlaceLid(
    above=None,
    on=None,
    x=None,
    y=None,
    z=None,
    gripper="close",
    useTarget=True
):
    if useTarget:
        on_target = getTarget(on)
        above_target = getTarget(above)

        # Faste targets bruger targetets egen frame
        useTargetFrame(on_target)
        robot.setSpeed(800, 480, 4000, 400)
        robot.MoveL(on_target)

        if gripper == "close":
            closeGripper()
        else:
            openGripper()

        useTargetFrame(above_target)
        robot.setSpeed(200, 120, 1000, 80)
        robot.MoveL(above_target)

    else:
        # Modulkoordinater bruger World frame
        useWorldFrame()

        above_target = makeModuleTarget(
            "Temp module above",
            x,
            y,
            z + MODULE_APPROACH_HEIGHT
        )

        on_target = makeModuleTarget(
            "Temp module on",
            x,
            y,
            z
        )

        robot.MoveL(on_target)

        if gripper == "close":
            closeGripper()
        else:
            openGripper()

        robot.MoveL(above_target)