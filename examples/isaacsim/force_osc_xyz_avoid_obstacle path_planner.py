"""
Move the UR5 CoppeliaSim arm to a target position while avoiding an obstacle.
The simulation ends after 1500 time steps, and the trajectory
of the end-effector is plotted in 3D.
"""
import numpy as np
from flying_cube import FlyingCube
from abr_control.arms import ur5 as arm
from abr_control.controllers import Joint, path_planners
from abr_control.utils import transformations



from abr_control.controllers import OSC, AvoidObstacles, Damping
from abr_control.interfaces.nv_isaacsim import IsaacSim

# initialize our robot config
robot_config = arm.Config()
# change this flag to False to use position control
use_force_control = False
n_timesteps = 2000
path_planner = path_planners.InverseKinematics(robot_config)



avoid = AvoidObstacles(robot_config)
# damp the movements of the arm
if use_force_control:
        # create an operational space controller
    damping = Damping(robot_config, kv=10)
    # instantiate the REACH controller with obstacle avoidance
    ctrlr = OSC(
        robot_config,
        kp=200,
        null_controllers=[avoid, damping],
        vmax=[0.5, 0],  # [m/s, rad/s]
        # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
        ctrlr_dof=[True, True, True, False, False, False],
    )

# create our CoppeliaSim interface
interface = IsaacSim(robot_config, dt=0.005)
interface.connect()

# set up lists for tracking data
ee_track = []
target_track = []
obstacle_track = []

moving_obstacle = True
isaac_obstacle = FlyingCube(interface.dt, interface.world, name ="obstacle", prim_path="/World/obstacle", color=np.array([0, 1.0, 0]))
isaac_target = FlyingCube(interface.dt, interface.world, name="target", prim_path="/World/target")
obstacle_xyz = np.array([0.09596, -0.2661, 0.64204])
try:
    # get visual position of end point of object
    feedback = interface.get_feedback()
    start = robot_config.Tx("EE", q=feedback["q"])
    # make the target offset from that start position
    target_xyz = start + np.array([0.2, -0.2, 0.0])
    interface.set_xyz(name="target", xyz=target_xyz)
    interface.set_xyz(name="obstacle", xyz=obstacle_xyz)


    count = 0.0
    obs_count = 0.0
    print("\nSimulation starting...\n")
    
    while count < n_timesteps :
        # get joint angle and velocity feedback
        feedback = interface.get_feedback()
        #target = np.hstack(
        #    [interface.get_xyz("target"), interface.get_orientation("target")]
        #)

        # position control
        R = robot_config.R("EE", q=feedback["q"])
        target_orientation = transformations.euler_from_matrix(R, "sxyz")
        # can use 3 different methods to calculate inverse kinematics
        # see inverse_kinematics.py file for details
        path_planner.generate_path(
            position=feedback["q"],
            target_position=np.hstack((   
                [interface.get_xyz("target"), interface.get_orientation("target")])),
            method=3,
            plot=False,
            )
        # returns desired [position, velocity]
        target, _ = path_planner.next()


        
        # get obstacle position from IsaacSim
        obs_x, obs_y, obs_z = interface.get_xyz("obstacle")  # pylint: disable=W0632
        # update avoidance system about obstacle position
        avoid.set_obstacles([[obs_x, obs_y, obs_z, 0.05]])
        if moving_obstacle is True:
            obs_x = 0.125 + 0.25 * np.sin(obs_count)
            obs_count += 0.05
            interface.set_xyz(name="obstacle", xyz=[obs_x, obs_y, obs_z])

        if use_force_control:
            # calculate the control signal
            u = ctrlr.generate(
                q=feedback["q"],
                dq=feedback["dq"],
                target=target,
            )
            # send forces into IsaacSim, step the sim forward
            interface.send_forces(u)
        else:
            # use position control, send angles into IsaacSim
            interface.send_target_angles(target[: robot_config.N_JOINTS])

          


            #print( 'robot_config: ' , robot_config)
            #print( 'robot_config.N_JOINTS: ' , robot_config.N_JOINTS)
            #print( 'target[: robot_config.N_JOINTS]: ' , target[: robot_config.N_JOINTS])


        # calculate end-effector position
        ee_xyz = robot_config.Tx("EE", q=feedback["q"])
        # track data
        ee_track.append(np.copy(ee_xyz))
        target_track.append(np.copy(target[:3]))
        obstacle_track.append(np.copy([obs_x, obs_y, obs_z]))

        count += 1

finally:
    # stop and reset IsaacSim 
    interface.disconnect()

    print("Simulation terminated...")

    ee_track = np.array(ee_track)
    target_track = np.array(target_track)
    obstacle_track = np.array(obstacle_track)

    if ee_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611

        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(211)
        ax1.set_ylabel("Distance (m)")
        ax1.set_xlabel("Time (ms)")
        ax1.set_title("Distance to target")
        ax1.plot(
            np.sqrt(np.sum((np.array(target_track) - np.array(ee_track)) ** 2, axis=1))
        )

        ax2 = fig.add_subplot(212, projection="3d")
        ax2.set_title("End-Effector Trajectory")
        ax2.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label="ee_xyz")
        ax2.scatter(
            target_track[0, 0],
            target_track[0, 1],
            target_track[0, 2],
            label="target",
            c="g",
        )
        ax2.plot(
            obstacle_track[:, 0],
            obstacle_track[:, 1],
            target_track[:, 2],
            label="obstacle",
            c="r",
        )
        ax2.legend()
        plt.show()
