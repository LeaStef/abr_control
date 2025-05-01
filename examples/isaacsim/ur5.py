import numpy as np
from abr_control.arms import ur5 as arm
from abr_control.interfaces.nv_isaacsim import IsaacSim
from abr_control.utils import transformations

# Sim step size
dt = 0.005

# Initialize our robot config
robot_config = arm.Config()

# Create our interface
interface = IsaacSim(robot_config, dt=dt)
interface.connect()
feedback = interface.get_feedback()
print ("q : " + str(feedback["q"]))
print ("dq : " + str(feedback["dq"]))

count = 0

for i in range(5000):
    val = count % 6
    interface.send_target_angles(np.array([val, 5.6,1.0,0.0,0.0,0.0]))

    count += 1
    import time
    time.sleep(5) 


interface.disconnect()