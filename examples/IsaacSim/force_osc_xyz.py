"""
Move the jaco2 IsaacSim arm to a target position.
The simulation ends after 1500 time steps, and the
trajectory of the end-effector is plotted in 3D.
"""
import sys
import traceback
import numpy as np
from abr_control.arms.isaacsim_config import IsaacsimConfig as arm
from abr_control.controllers import OSC, Damping
from abr_control.interfaces.nv_isaacsim import IsaacSim
from abr_control.utils import transformations
import time
last_time = time.time()


if len(sys.argv) > 1:
    arm_model = sys.argv[1]
else:
    arm_model = "jaco2"
robot_config = arm(arm_model)

# create our IsaacSim interface
dt = 0.007  # 143 Hz 
#dt = 0.001 # 1000 Hz

target_prim_path="/World/target"
interface = IsaacSim(robot_config, dt=dt)
interface.connect(joint_names=[f"joint{ii}" for ii in range(len(robot_config.START_ANGLES))])

interface.send_target_angles(robot_config.START_ANGLES)
isaac_target = interface.create_target_prim(prim_path=target_prim_path)
interface.fix_arm_joint_gains()

print("joint_pos_addrs: ", interface.robot_config.joint_pos_addrs)


# damp the movements of the arm
damping = Damping(robot_config, kv=30)  #kv=10)
# instantiate controller
ctrlr = OSC(
    robot_config,
    kp= 300, # 200
    null_controllers=[damping],
    vmax=[0.5, 0],  # [m/s, rad/s]
    # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[True, True, True, False, False, False],
)

# set up lists for tracking data
ee_track = []
target_track = []

#target_geom = "target"
green = [0, 0.9, 0, 0.5]
red = [0.9, 0, 0, 0.5]

np.random.seed(0)
def gen_target(interface):
    target_xyz = (np.random.rand(3) + np.array([-0.5, -0.5, 0.5])) * np.array(
        [1, 1, 0.5]
    )
    interface.set_xyz(target_prim_path, target_xyz)

try:
    # get the end-effector's initial position
    feedback = interface.get_feedback()
    start = robot_config.Tx(interface.ee_link_name, feedback["q"])
    
    # make the target offset from that start position
    gen_target(interface)

    count = 0
    debug = 0
    print("\nSimulation starting...\n")
    while 1:
        current_time = time.time()
        dt = current_time - last_time
        print(f"Control dt: {dt:.6f}s, Freq: {1/dt:.1f}Hz")
        last_time = current_time
        # get joint angle and velocity feedback
        feedback = interface.get_feedback()

        target = np.hstack(
            [
                interface.get_xyz(target_prim_path),
                transformations.euler_from_quaternion(
                    interface.get_orientation(target_prim_path), "rxyz"
                ),
            ]
        )

        # calculate the control signal
        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
        )
        

        interface.send_forces(u)

        # calculate end-effector position
        ee_xyz = robot_config.Tx(interface.ee_link_name, q=feedback["q"])
        # track data
        ee_track.append(np.copy(ee_xyz))
        target_track.append(np.copy(target[:3]))

        error = np.linalg.norm(ee_xyz - target[:3])
        if error < 0.02:
            # interface.model.geom(target_geom).rgba = green
            count += 1
        else:
            count = 0
            # interface.model.geom(target_geom).rgba = red

        if count >= 50:
            print("Generating a new target")
            gen_target(interface)
            count = 0

except:
    print(traceback.format_exc())

finally:
    # stop and reset the IsaacSim simulation
    interface.disconnect()

    print("Simulation terminated...")

    ee_track = np.array(ee_track)
    target_track = np.array(target_track)

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
            target_track[1, 0],
            target_track[1, 1],
            target_track[1, 2],
            label="target",
            c="r",
        )
        ax2.scatter(
            ee_track[0, 0],
            ee_track[0, 1],
            ee_track[0, 2],
            label="start",
            c="g",
        )
        ax2.legend()
        plt.show()
