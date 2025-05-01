"""
Running the joint controller with an inverse kinematics path planner
in IsaacSim. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which can be moved by the user.
"""

import numpy as np
from abr_control.arms import ur5 as arm
from abr_control.controllers import Joint, path_planners

from abr_control.interfaces.nv_isaacsim import IsaacSim
from flying_cube import FlyingCube
from abr_control.utils import transformations


if __name__ == "__main__":
    dt = 0.001

    # change this flag to False to use position control
    use_force_control = False
    # Initialize our robot config
    robot_config = arm.Config()
    if use_force_control:
        # create an operational space controller
        ctrlr = Joint(robot_config, kp=300, kv=20)

    # create our path planner
    n_timesteps = 2000
    path_planner = path_planners.InverseKinematics(robot_config)

    # Create our interface
    interface = IsaacSim(robot_config, dt=dt)
    interface.connect()

    flying_cube = FlyingCube(dt, interface.world)
    feedback = interface.get_feedback()
    print ("q : " + str(feedback["q"]))
    print ("dq : " + str(feedback["dq"]))

    count = 0
    for t in np.arange(0, 100, dt):
        # get arm feedback
        #velocity = np.array([np.sin(t), np.cos(t), 0.])
        #state = flying_cube.step(velocity)

        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx("EE", feedback["q"])
       
        if count % n_timesteps == 0:
            # get pos of flying cube
            target_xyz = flying_cube._get_obj_pos()
            # print('target_xyz ', target_xyz)

            R = robot_config.R("EE", q=feedback["q"])
            target_orientation = transformations.euler_from_matrix(R, "sxyz")
            
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
        target, _ = path_planner.next()
        if use_force_control:
            # generate an operational space control signal
            u = ctrlr.generate(
                q=feedback["q"],
                dq=feedback["dq"],
                target=target,
            )

            # apply the control signal, step the sim forward
            interface.send_forces(u)

        else:
            # use position control
            interface.send_target_angles(target[: robot_config.N_JOINTS])

        count += 1
    # stop and reset the simulation
    interface.disconnect()
    print("Simulation terminated...")