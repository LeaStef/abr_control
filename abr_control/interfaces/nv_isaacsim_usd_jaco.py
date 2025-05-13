import numpy as np
from isaacsim import SimulationApp
from .interface import Interface

class IsaacSim(Interface):
    """An interface for IsaacSim.

    Parameters
    ----------
    robot_config : class instance
        contains all relevant information about the arm
        such as: number of joints, number of links, mass information etc.
    dt : float, optional (Default: 0.001)
        simulation time step in seconds

    """
    def __init__(self, robot_config, dt=0.001, force_download=False):

        super().__init__(robot_config)
        self.dt = dt  # time step
        self.count = 0  # keep track of how many times send forces is called

        self.q = np.zeros(self.robot_config.N_JOINTS)  # joint angles
        self.dq = np.zeros(self.robot_config.N_JOINTS)  # joint_velocities

        self.prim_path = "/World/robot"
        self.name = "robot"
      
        #self.misc_handles = {}  # for tracking miscellaneous object handles
        

    def connect(self, load_scene=True):
        if load_scene:
            """All initial setup."""
            self.simulation_app = SimulationApp({"headless": False}) 
            from isaacsim.core.api import World # type: ignore
            import omni.isaac.core.utils.stage as stage_utils  # type: ignore   
            import omni.kit.commands # type: ignore
            import omni
            from isaacsim.storage.native import get_assets_root_path
            from isaacsim.core.api.robots import Robot
            
            # Create a world
            self.world = World(physics_dt=self.dt,rendering_dt=self.dt)
            self.world.scene.add_default_ground_plane()
            
            assets_root_path = get_assets_root_path()
            
            stage_utils.add_reference_to_stage(
                 usd_path=assets_root_path + "/Isaac/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd",
                 # Robots/Kinova/Jaco2/J2N7S300/j2n7s300_instanceable.usd   -->  7 DOF arm , not compatible with ABR controller
                 prim_path=self.prim_path,
                 )

            robot = self.world.scene.add(Robot(prim_path=self.prim_path, name=self.name))


        else:
            self.world = SimulationApp.getWorld()
            
        # Get the articulation
        from omni.isaac.core.articulations import Articulation# type: ignore
       
        # Resetting the world needs to be called before querying anything related to an articulation specifically.
        # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
        self.world.reset()
        # Load robot
        self.articulation = Articulation(prim_path=self.prim_path, name="H1")
        self.articulation.initialize()
        print("DOF names:", self.articulation.dof_names)
        
        # necessary so self.q and self.dq are accessible
        self.world.initialize_physics()



    def disconnect(self):
        """Any socket closing etc that must be done to properly shut down"""
        self.simulation_app.close() # close Isaac Sim
        print("IsaacSim connection closed...")


    def send_forces(self, u):
        """Applies the set of torques u to the arm. If interfacing to
        a simulation, also moves dynamics forward one time step.

        u : np.array
            An array of joint torques [Nm]
        """
        print("robot joint pos: ", self.robot.get_joint_positions())
        print("u: ", u)
        # Apply the control signal
        #self.articulation.set_joint_efforts(u)
        #self.robot.set_joint_efforts(u)
        self.robot.set_joint_efforts(u)

         # move simulation ahead one time step
        self.world.step(render=True) # execute one physics step and one rendering step



    def send_target_angles(self, q):
        """Moves the arm to the specified joint angles

        q : numpy.array
            the target joint angles [radians]
        """
        if np.size(q) == 12:
            self.articulation.set_joint_positions(q)
        if np.size(q) == 6:
            fb = self.get_feedback(with_hand=True)
            fb_q = fb["q"]
            fb_q[:6] = q
            self.articulation.set_joint_positions(fb_q)
        else:
            print("The method send_target_angles of the isaacsim interface does not support the number of joint angles that are attempted to be set.")
        
        # move simulation ahead one time step
        self.world.step(render=True) # execute one physics step and one rendering step



    def get_feedback(self, with_hand=False):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """
        q = self.articulation.get_joint_positions()
        dq = self.articulation.get_joint_velocities()
        if with_hand:
            # Get the joint angles and velocities
            self.q = q
            self.dq = dq
        else:
            self.q = q[:6]
            self.dq = dq[:6]
        return {"q": self.q, "dq": self.dq}
    


    def get_xyz(self, name):
                """Returns the xyz position of the specified object

                name : string
                    name of the object you want the xyz position of
                """
                print("NAME: ", name)
                #TODO check if we need misc handles for this
                obj = self.world.scene.get_object(name) 
                object_position, object_orientation = obj.get_world_pose()

                return object_position
    

    def get_orientation(self, name):
        """Returns the orientation of an object in CoppeliaSim

        the Euler angles [radians] are returned in the relative xyz frame.
        http://www.coppeliarobotics.com/helpFiles/en/eulerAngles.htm

        Parameters
        ----------
        name : string
            the name of the object of interest
        """

        obj = self.world.scene.get_object(name) 
        object_position, object_orientation = obj.get_world_pose()
        return object_orientation


    def set_xyz(self, name, xyz):
        """Set the position of an object in the environment.

        name : string
            the name of the object
        xyz : np.array
            the [x,y,z] location of the target [meters]
        """
        _cube = self.world.scene.get_object(name)
        _cube.set_world_pose(xyz, np.array([0., 0., 0., 1.])) # set the position and orientation of the object
