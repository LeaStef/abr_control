import numpy as np
from abr_control.arms import ur5 as arm
from abr_control.interfaces.nv_isaacsim import IsaacSim
from flying_cube import FlyingCube

if __name__ == "__main__":
    dt = 0.01

    # Initialize our robot config
    robot_config = arm.Config()

    # Create our interface
    interface = IsaacSim(robot_config, dt=dt)
    interface.connect()

    flying_cube = FlyingCube(dt, interface.world)

    interface.send_target_angles(np.array([1.0, 0.0,2.0,3.0,0.0,0.0]))
    feedback = interface.get_feedback()
    print ("q : " + str(feedback["q"]))
    print ("dq : " + str(feedback["dq"]))

    # Example: Apply sinusoidal velocity to the cube
    for t in np.arange(0, 10, dt):
        velocity = np.array([np.sin(t), np.cos(t), 0.])
        state = flying_cube.step(velocity)
        #print(f"Time: {t:.2f}, State: {state}")