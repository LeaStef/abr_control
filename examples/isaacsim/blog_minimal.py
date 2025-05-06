import numpy as np
from abr_control.arms import ur5 as arm
from abr_control.controllers import OSC
from abr_control.interfaces.nv_isaacsim import IsaacSim
from flying_cube import FlyingCube

# https://studywolf.wordpress.com/2017/07/01/abr-control-0-1-repo-public-release/ 


# initialize our robot config for the ur5
robot_config = arm.Config(use_cython=True)
 
# instantiate controller
ctrlr = OSC(robot_config, kp=200, vmax=[0.5, 0])

 
# create our VREP interface
interface = IsaacSim(robot_config, dt=.001)
interface.connect()
 
isaac_target = FlyingCube(interface.dt, interface.world, name="target", prim_path="/World/target")
target_xyz = np.array([0.2, 0.2, 0.2])
# set the target object's position in VREP
interface.set_xyz(name='target', xyz=target_xyz)
 
count = 0.0
while count < 4000:  # run for 1.5 simulated seconds
    # get joint angle and velocity feedback
    feedback = interface.get_feedback()
    # calculate the control signal
    u = ctrlr.generate(
        q=feedback['q'],
        dq=feedback['dq'],
        target=target_xyz)
    
   
    # send forces into VREP, step the sim forward
    interface.send_forces(u)
 
    count += 1
interface.disconnect()