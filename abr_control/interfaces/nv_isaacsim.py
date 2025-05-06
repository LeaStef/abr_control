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
    def __init__(self, robot_config, dt=0.001):

        super().__init__(robot_config)

        self.dt = dt  # time step
        self.count = 0  # keep track of how many times send forces is called

        self.q = np.zeros(self.robot_config.N_JOINTS)  # joint angles
        self.dq = np.zeros(self.robot_config.N_JOINTS)  # joint_velocities

        self.import_config = None
        self.world = None
        self.prim_path = None
        self.robot = None
        self.ai_link = None
        #self.misc_handles = {}  # for tracking miscellaneous object handles

        '''
        # joint target velocities, as part of the torque limiting control
        # these need to be super high so that the joints are always moving
        # at the maximum allowed torque
        self.joint_target_velocities = np.ones(robot_config.N_JOINTS) * 10000.0
        '''
        

    def connect(self, load_scene=True):
        if load_scene:
            """All initial setup."""
            self.simulation_app = SimulationApp({"headless": False}) 
            from isaacsim.core.api import World # type: ignore
            from isaacsim.asset.importer.urdf import _urdf # type: ignore
            from omni.isaac.core.robots.robot import Robot # type: ignore
            import omni.kit.commands # type: ignore
            import omni
            # Create a world
            self.world = World(physics_dt=self.dt,rendering_dt=self.dt)
            self.world.scene.add_default_ground_plane()

            # Acquire the URDF extension interface for parsing and importing URDF files
            urdf_interface = _urdf.acquire_urdf_interface()

            # Configure the settings for importing the URDF file
            self.import_config = _urdf.ImportConfig()
            self.import_config.convex_decomp = False  # Disable convex decomposition for simplicity
            self.import_config.fix_base = True       # Fix the base of the robot to the ground
            self.import_config.make_default_prim = True  # Make the robot the default prim in the scene
            self.import_config.self_collision = False  # Disable self-collision for performance
            self.import_config.distance_scale = 1     # Set distance scale for the robot
            self.import_config.density = 0.0          # Set density to 0 (use default values)

            # Set the path to the URDF file
            root_path = '/home/steffen/workspace_common/ur_description/urdf/'
            file_name = 'ur5.urdf'
            imported_robot =  urdf_interface.parse_urdf(root_path, file_name, self.import_config)
            self.prim_path = urdf_interface.import_robot(root_path, file_name, imported_robot, self.import_config, "ur5")
            self.robot = Robot(self.prim_path)
            self.ai_link = self.world.scene.add(self.robot)
        else:
            self.world = SimulationApp.getWorld()
            
        # Get the articulation
        from omni.isaac.core.articulations import Articulation, ArticulationSubset # type: ignore
        import omni.isaac.core.utils.stage as stage_utils # type: ignore
       
        # Resetting the world needs to be called before querying anything related to an articulation specifically.
        # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
        self.world.reset()
        # necessary so self.q and self.dq are accessible
        self.world.initialize_physics()

        # Load robot
        self.articulation = Articulation(prim_path=self.prim_path, name="ur5")
        self.articulation.initialize()
        
        print("Started Isaacsim as stand alone app...")



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
        print("robot joint pos: ", self.robot.get_joint_positions())
        print("q: ", q)
        self.robot.set_joint_positions(q)
        
        # move simulation ahead one time step
        self.world.step(render=True) # execute one physics step and one rendering step



    def get_feedback(self):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """
        # Get the joint angles and velocities
        self.q = self.articulation.get_joint_positions()
        self.dq = self.articulation.get_joint_velocities()
        return {"q": self.q, "dq": self.dq}


    def get_xyz(self, name):
                """Returns the xyz position of the specified object

                name : string
                    name of the object you want the xyz position of
                """
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
