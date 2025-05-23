"""
Running the joint controller with an inverse kinematics path planner
for a Mujoco simulation. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which changes every n time steps.
"""
import numpy as np
#from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.arms import unitree_H1 as arm
#import glfw
from abr_control.controllers import path_planners
#from abr_control.interfaces.mujoco import Mujoco
from abr_control.interfaces.nv_isaacsim import IsaacSim
from abr_control.utils import transformations
from flying_cube import FlyingCube

'''
from abr_control.arms import ur5 as arm1
ur5 = arm1.Config()
print("ur5: ", ur5.N_JOINTS)
print("L: ", len(ur5.L))
from abr_control.arms import jaco2 as arm2
jaco = arm2.Config()
print("jaco: ", jaco.N_JOINTS)
print("L: ", len(jaco.L))

from abr_control.arms import onejoint as arm3
onejoint = arm3.Config()
print("onejoint: ", onejoint.N_JOINTS)
print("L: ", len(onejoint.L))

from abr_control.arms import twojoint as arm4
twojoint = arm4.Config()
print("twojoint: ", twojoint.N_JOINTS)

print("L: ", len(twojoint.L))

from abr_control.arms import threejoint as arm5
threejoint = arm5.Config()
print("threejoint: ", threejoint.N_JOINTS)
print("L: ", len(threejoint.L))
'''

robot_config = arm.Config()
#print("h1: ", robot_config.N_JOINTS)
#print("L: ", len(robot_config.L))


# create our path planner
n_timesteps = 2000
path_planner = path_planners.InverseKinematics(robot_config)

# create our interface
dt = 0.001
interface = IsaacSim(robot_config, dt=dt)
interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)
feedback = interface.get_feedback()
isaac_target = FlyingCube(interface.dt, interface.world, name="target", prim_path="/World/target")




#inertias = interface.articulation_view.get_body_inertias()

#inertias_right_shoulder_pitch_joint = interface.articulation_view.get_body_inertias(body_indices=np.array([6]))


try:
    print("\nSimulation starting...")
    print("Click to move the target.\n")

    count = 0
    while 1:
       
        if count % n_timesteps == 0:
            print("Moving target...")
            feedback = interface.get_feedback()
            target_xyz = np.array(
                [
                    np.random.random() * 0.5 + 0.2,  
                    np.random.random() * 1.1 - 0.9,
                    np.random.random() * 1.0 + 0.9,
                ]
            )
            R = robot_config.R("EE", q=feedback["q"])
            #R = robot_config.R("wrist_3_joint", q=feedback["q"])
            
            target_orientation = transformations.euler_from_matrix(R, "sxyz")
            # update the position of the target
            interface.set_xyz("target", target_xyz)

            # can use 3 different methods to calculate inverse kinematics
            # see inverse_kinematics.py file for details
            path_planner.generate_path(
                position=feedback["q"],
                target_position=np.hstack([target_xyz, target_orientation]),
                method=3,
                dt=0.005,
                n_timesteps=n_timesteps,
                plot=False,
            )

        # returns desired [position, velocity]
        target = path_planner.next()[0]

        # use position control
        interface.send_target_angles(target[: robot_config.N_JOINTS])
        #interface.world.step(render=True) # execute one physics step and one rendering step
        
        count += 1

finally:
    # stop and reset the simulation
    interface.disconnect()

    print("Simulation terminated...")