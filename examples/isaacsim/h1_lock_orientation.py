"""
Running the joint controller with an inverse kinematics path planner
for a Mujoco simulation. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which changes every n time steps.
"""
import numpy as np
#from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.arms import jaco2 as arm
from abr_control.interfaces.nv_isaacsim import IsaacSim


robot_config = arm.Config()

# create our path planner
n_timesteps = 2000
# create our interface
dt = 0.001
interface = IsaacSim(robot_config, dt=dt)
interface.connect()


try:
    print("\nSimulation starting...")
    count = 0
    while 1:
        
        feedback = interface.get_feedback(arm_only=False)
        q=feedback["q"]
        #print("Q: ", q)

        interface.send_target_angles(q)
        count += 1


finally:
    # stop and reset the simulation
    interface.disconnect()

    print("Simulation terminated...")