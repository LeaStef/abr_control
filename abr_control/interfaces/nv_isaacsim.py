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

        # ee_name = "ft_frame" for ur2 
        # #"end_effector"  for jaco2

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
        '''
        # setting up import configuration:
        status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
        import_config.set_fix_base(True)  # fix the base of the robot
        import_config.set_make_default_prim(False)

        # Get path to extension data:
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.mjcf")
        extension_path = ext_manager.get_extension_path(ext_id)
        # import MJCF
        omni.kit.commands.execute(
            "MJCFCreateAsset", 
            #mjcf_path=extension_path + "/data/mjcf/nv_ant.xml",
            mjcf_path=extension_path + "/data/mjcf/nv_humanoid.xml",
            #mjcf_path=self.robot_config.xml_file,
            import_config=import_config,
            prim_path=self.prim_path
        )
        '''
        
        
        if self.robot_config.robot == "ur5":
            robot_path = "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
            self.ee_name = "flange" #"ft_frame" # end-effector name for UR5
        elif self.robot_config.robot == "jaco2":
            robot_path = "/Isaac/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd"
            self.ee_name = "end_effector"  # end-effector name for Jaco2
        elif self.robot_config.robot == "h1":   
            robot_path = "/Isaac/Robots/Unitree/H1/h1.usd"
            #TODO check this
            self.ee_name = "EE"  # end-effector name for H1


        assets_root_path = get_assets_root_path()
        stage_utils.add_reference_to_stage(
                 usd_path=assets_root_path + robot_path,
                 prim_path=self.prim_path,
                 )
        robot = self.world.scene.add(Robot(prim_path=self.prim_path, name=self.name))


        '''
        ## LOAD UR5 robot
        assets_root_path = get_assets_root_path()
        stage_utils.add_reference_to_stage(
                 usd_path=assets_root_path + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
                 # Robots/Kinova/Jaco2/J2N7S300/j2n7s300_instanceable.usd   -->  7 DOF arm , not compatible with ABR controller
                 prim_path=self.prim_path,
                 )
        robot = self.world.scene.add(Robot(prim_path=self.prim_path, name=self.name))

        
        ## LOAD Jaco2 robot
        assets_root_path = get_assets_root_path()
        stage_utils.add_reference_to_stage(
                 usd_path=assets_root_path + "/Isaac/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd",
                 # Robots/Kinova/Jaco2/J2N7S300/j2n7s300_instanceable.usd   -->  7 DOF arm , not compatible with ABR controller
                 prim_path=self.prim_path,
                 )
        robot = self.world.scene.add(Robot(prim_path=self.prim_path, name=self.name))
        
        
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
         
        self.world.step(render=False)

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
            self.prim_path,
            self.ee_name,
            joint_names
        )
        

    def disconnect(self):
        """Any socket closing etc that must be done to properly shut down"""
        self.simulation_app.close() # close Isaac Sim
        print("IsaacSim connection closed...")

    #TODO adapt to IsaacSim
    def get_joint_pos_addrs(self, jntadr):
        # store the data.qpos indices associated with this joint
        first_pos = self.model.jnt_qposadr[jntadr]
        posvec_length = self.robot_config.JNT_POS_LENGTH[self.model.jnt_type[jntadr]]
        joint_pos_addr = list(range(first_pos, first_pos + posvec_length))[::-1]
        return joint_pos_addr


    def get_joint_vel_addrs(self, joint_name):
        if self.articulation_view is None:
            raise RuntimeError("Robot ArticulationView not set up.")
        
        #dof_indices = self.articulation_view.get_dof_indices(joint_name)
        #print('dof_indices:', dof_indices)
         # Get all DOF names in the articulation
        dof_names = self.articulation_view.dof_names
        dof_name_to_index = {name: i for i, name in enumerate(dof_names)}
        
        if joint_name.endswith(self.ee_name):
            index = None
        else:
            index = dof_name_to_index[joint_name]

        return index
    '''   
    def send_forces(self, u):
        """Applies the set of torques u to the arm."""
        
        # Create full torque vector for all DOFs
        total_dofs = self.articulation_view.num_dof
        full_torques = np.zeros(total_dofs)
        
        # Apply control torques to the controlled joints
        full_torques[:len(u)] = u
        
        # Apply the control signal
        self.articulation_view.set_joint_efforts(full_torques)
        
        # Move simulation ahead one time step
        self.world.step(render=True)
        
    '''
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
        
        print("q: ", q)
        print("robot joint pos: ", self.articulation.get_joint_positions())
        q_all = self.get_feedback()["q"]
        print("q_all: ", q_all)
        #TODO change that to variable number of joints
        #q_all = q_all[:self.robot_config.N_JOINTS]  #
        q_all[:6] = q
        print("result: ", q_all)
        self.articulation_view.set_joint_positions(q_all)
        """
        print("len(q) : ", len(q))
        print("robot_config.N_JOINTS: ", self.robot_config.N_JOINTS)
        # Check if the length of q is greater than the number of joints
        if len(q) > self.robot_config.N_JOINTS:
            q_new = q[:self.robot_config.N_JOINTS]  
            self.articulation_view.set_joint_positions(q_new)
        elif len(q) < self.robot_config.N_ALL_JOINTS :
            q_new = self.articulation_view.get_joint_positions()  # Shape: (1, 12)
            q_new[0, :len(q)] = q  # Update first N_JOINTS for environment 0
            self.articulation_view.set_joint_positions(q_new)
        else: 
            self.articulation_view.set_joint_positions(q)
      
        '''
        if len(q) > self.robot_config.N_JOINTS:
            q_new = q[:self.robot_config.N_JOINTS]  
            self.articulation_view.set_joint_positions(q_new)
        elif self.robot_config.N_ALL_JOINTS > self.robot_config.N_JOINTS:
            q_new = self.articulation_view.get_joint_positions()  # Shape: (1, 12)
            q_new[0, :self.robot_config.N_JOINTS] = q  # Update first N_JOINTS for environment 0
            self.articulation_view.set_joint_positions(q_new)
        else: 
            self.articulation_view.set_joint_positions(q)
        '''
      


    
    def get_feedback(self, all_joints=False):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """
        if not all_joints:
            # Get the joint angles and velocities
            self.q = self.articulation.get_joint_positions()[:self.robot_config.N_JOINTS]  # only take the first N_JOINTS
            self.dq = self.articulation.get_joint_velocities()[:self.robot_config.N_JOINTS] 
        else:
            # Get the joint angles and velocities for all joints
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
    
    #TODO check if overlap to def quaternion
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
    

    #TODO change method name in 'set_named_prim' or something as mocap is mujoco thing
    def set_mocap_xyz(self, name, xyz):
        """
        Set the world position of a named prim (used like a mocap target).

        Parameters
        ----------
        name : str
            Name of the prim (e.g. site or target object)
        xyz : np.ndarray
            Target world position [x, y, z] in meters
        """
       
        world_path = "/World"
        prim_path = f"{world_path}/{name}" 
        print("prim_path:", prim_path)
        prim = self.stage.GetPrimAtPath(prim_path)

        prim.set_world_pose(xyz, np.array([0., 0., 0., 1.])) # set the position and orientation of the object


        print("prim_path:", prim_path)
        #from omni.isaac.core.utils.prims import set_prim_world_position
        #set_prim_world_position(prim_path, xyz)




    #TODO remove as is the same as above
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



    #TODO use for R and Tx in isaacsim_config
    def get_prim_ends_with_name(self, name):
        prim_path = None
        for prim in self.stage.Traverse():
            print("prim: ", prim)
            print("prim.GetPath(): ", prim.GetPath())
            print("name: ", name)
            if str(prim.GetPath()).endswith(name):
                prim_path = prim.GetPath()
                break
        return prim_path
    