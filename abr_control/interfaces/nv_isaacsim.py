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
        self.robot_config = robot_config
        self.dt = dt  # time step
        self.count = 0  # keep track of how many times send forces is called

        #self.q = np.zeros(self.robot_config.N_JOINTS)  # joint angles
        #self.initial_q = np.zeros(self.robot_config.N_JOINTS)  # joint angles
        #self.dq = np.zeros(self.robot_config.N_JOINTS)  # joint_velocities

        self.prim_path = "/World/robot"
        self.name = self.robot_config.xml_dir.rsplit('/', 1)[-1]
        #self.name = "robot"
        #robot_name = self.robot_config.xml_dir.rsplit('/', 1)[-1]

      
        #self.misc_handles = {}  # for tracking miscellaneous object handles
        

    def connect(self, joint_names=None):
        """
        joint_names: list, optional (Default: None)
            list of joint names to send control signal to and get feedback from
            if None, the joints in the kinematic tree connecting the end-effector
            to the world are used
        """

      
        """All initial setup."""
        self.simulation_app = SimulationApp({"headless": False}) 
        from isaacsim.core.api import World # type: ignore
        import omni.isaac.core.utils.stage as stage_utils  # type: ignore   
        import omni.kit.commands # type: ignore
        import omni
        from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
        from isaacsim.storage.native import get_assets_root_path
        from omni.isaac.core.articulations import Articulation, ArticulationView # type: ignore
        import omni.isaac.core.utils.stage as stage_utils # type: ignore
        from isaacsim.core.api.robots import Robot
        from pxr import UsdLux, Sdf, Gf, UsdPhysics
        # Create a world
        self.world = World(physics_dt=self.dt,rendering_dt=self.dt)
        self.context = omni.usd.get_context()
        self.stage = self.context.get_stage()
        #TODO necessary for H1 robot
        #self.world.add_physics_callback("send_actions", self.send_actions)
        self.world.scene.add_default_ground_plane()

        # enable physics
        scene = UsdPhysics.Scene.Define(self.stage, Sdf.Path("/physicsScene"))

        # set gravity
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(981.0)

        # add lighting
        distantLight = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/DistantLight"))
        distantLight.CreateIntensityAttr(500)
        
        ## LOAD Jaco2 robot
        assets_root_path = get_assets_root_path()
        stage_utils.add_reference_to_stage(
                 usd_path=assets_root_path + "/Isaac/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd",
                 # Robots/Kinova/Jaco2/J2N7S300/j2n7s300_instanceable.usd   -->  7 DOF arm , not compatible with ABR controller
                 prim_path=self.prim_path,
                 )
        robot = self.world.scene.add(Robot(prim_path=self.prim_path, name=self.name))
        
        '''
        # Load H1 robot
        self.h1 = H1FlatTerrainPolicy(
            prim_path=self.prim_path,
            name=self.name,
            usd_path=assets_root_path + "/Isaac/Robots/Unitree/H1/h1.usd",
            position=np.array([0, 0, 1.05]),
        )
        stage_utils.add_reference_to_stage(
            usd_path=assets_root_path + "/Isaac/Robots/Unitree/H1/h1.usd",
            prim_path=self.prim_path,
        )
        '''      
        # Resetting the world needs to be called before querying anything related to an articulation specifically.
        # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
        self.world.reset()
        
        # Load robot
        self.articulation = Articulation(prim_path=self.prim_path, name=self.name + "_articulation")
        self.articulation.initialize()
        self.world.scene.add(self.articulation) # Add to scene if not already added by higher-level env

        self.articulation_view = ArticulationView(prim_paths_expr=self.prim_path, name=self.name + "_view")
        self.world.scene.add(self.articulation_view)
        self.articulation_view.initialize()
     
        self.world.reset()
        # necessary so self.q and self.dq are accessible
        self.world.initialize_physics()

        self.joint_pos_addrs = []
        self.joint_vel_addrs = []
        self.joint_dyn_addrs = []

        print("Connecting to robot config...")
        self.robot_config._connect(
            self.world,
            self.stage,
            self.articulation,
            self.articulation_view,
            self.joint_pos_addrs,
            self.joint_vel_addrs,
            self.prim_path
        )
        

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
        self.articulation_view.set_joint_efforts(u)

         # move simulation ahead one time step
        self.world.step(render=True) # execute one physics step and one rendering step


    def send_target_angles(self, q):
        """Moves the arm to the specified joint angles

        q : numpy.array
                the target joint angles [radians]
        """


    
    def get_feedback(self, arm_only=True):
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



    # method for keep_standing
    def send_actions(self, dt):
        pelvis_prim_path = '/World/robot/pelvis'
        from pxr import Gf  # type: ignore    
        prim=self.stage.GetPrimAtPath(pelvis_prim_path)
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0 ,0.0 ,0.0 ,0.0))
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.0 ,0.0 ,0.02))
        #prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0.70711 ,0.70711 ,0.0 ,0.0))
    