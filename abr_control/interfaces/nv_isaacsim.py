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
        



    def connect(self, joint_names=None, camera_id=-1):
        """
        joint_names: list, optional (Default: None)
            list of joint names to send control signal to and get feedback from
            if None, the joints in the kinematic tree connecting the end-effector
            to the world are used
        """

      
        """All initial setup."""
        self.simulation_app = SimulationApp({"headless": False}) 
        import omni
        import omni.isaac.core.utils.stage as stage_utils  # type: ignore   
        from omni.isaac.core import World
        #from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.articulations import Articulation, ArticulationView # type: ignore
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        #TODO change import 
        #TODO is "Robot" even necessary 
        from isaacsim.core.api.robots import Robot
        from pxr import UsdPhysics, UsdGeom
        
        # Initialize the simulation world
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.context = omni.usd.get_context()
        self.stage = self.context.get_stage()
         #TODO necessary for H1 robot
        #self.world.add_physics_callback("send_actions", self.send_actions)
        
        robot_path = None
        # Load the robot from USD file
        if self.robot_config.robot == "ur5":
            robot_path = "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
            self.ee_link_name = "flange" #"ft_frame" # end-effector name for UR5
        elif self.robot_config.robot == "jaco2":
            robot_path = "/Isaac/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd"
            self.ee_link_name = "end_effector"  # end-effector name for Jaco2
        elif self.robot_config.robot == "h1":   
            robot_path = "/Isaac/Robots/Unitree/H1/h1.usd"
            #TODO check this
            self.ee_link_name = "EE"  # end-effector name for H1


        assets_root_path = get_assets_root_path()
        robot_usd_path = f"{assets_root_path}{robot_path}"
        print(f"Loading robot from USD path: {robot_usd_path}")
        

        
        stage_utils.add_reference_to_stage(
                 usd_path=assets_root_path + robot_path,
                 prim_path=self.prim_path,
                 )
        robot = self.world.scene.add(Robot(prim_path=self.prim_path, name=self.name))
        '''
        self.robot = self.world.scene.add(
            Articulation(
                prim_path="/World/Robot",
                name="robot",
                usd_path=robot_usd_path,
                position=[0, 0, 0]
            )
        )
        '''
        self.world.reset()
        self.articulation = Articulation(prim_path=self.prim_path, name=self.name + "_articulation")
        self.articulation.initialize()
        self.world.scene.add(self.articulation) # Add to scene if not already added by higher-level env

        #TODO remove and replace with articulation
        self.articulation_view = ArticulationView(prim_paths_expr=self.prim_path, name=self.name + "_view")
        self.world.scene.add(self.articulation_view)
        self.articulation_view.initialize()
       
        
        # Set simulation time step
        self.world.get_physics_context().set_physics_dt(self.dt)
        
        # Reset the world to initialize physics
        self.world.reset()
        
        # Get joint information
        self.joint_pos_addrs = []
        self.joint_vel_addrs = []
        self.joint_dyn_addrs = []
        

        
        if joint_names is None:
            print("No joint names provided, using all controllable joints in the articulation.")
            # Get all controllable joints in the articulation
            joint_names = self.articulation.dof_names
            # If we need to filter to kinematic chain from EE to base
            #TODO folowing, never jumped into, maybe remove
            if hasattr(self.robot_config, 'ee_link_name'):
                print("Using end-effector link name from robot config:", self.robot_config.ee_link_name)
                ee_link_name = self.robot_config.ee_link_name
                # Get kinematic chain from end-effector to base
                joint_names = self._get_kinematic_chain_joints(ee_link_name)
        else:
            # Handle joint name mapping
            joint_names = self._map_joint_names(joint_names)



        # Validate joint names and get indices
        all_joint_names = self.articulation.dof_names
        self.controlled_joint_indices = []
        
        for name in joint_names:
            if name not in all_joint_names:
                raise Exception(f"Joint name {name} does not exist in robot model")
            joint_idx = all_joint_names.index(name)
            self.controlled_joint_indices.append(joint_idx)
            self.joint_pos_addrs.append(joint_idx)
            self.joint_vel_addrs.append(joint_idx)
            self.joint_dyn_addrs.append(joint_idx)
        
        # Store joint names for later use
        self.controlled_joint_names = joint_names
        
        # Initialize joint position and velocity arrays
        self.num_dof = len(self.controlled_joint_indices)

        
        # Connect robot config with simulation data
        print("Connecting to robot config...")
        self.robot_config._connect(
            self.world,
            self.stage,
            self.articulation,
            self.articulation_view,
            self.joint_pos_addrs,
            self.joint_vel_addrs,
            self.prim_path,
            self.ee_link_name,
        )
        

        print(f"Connected to robot with {self.num_dof} controlled joints: {self.controlled_joint_names}")




    '''
    def _get_kinematic_chain_joints(self, ee_link_name):
        """
        Get joints in kinematic chain from end-effector to base
        """
        
        # For most robot arms, we can use all revolute joints
        # This would need to be customized based on your specific robot
        all_joints = self.articulation.dof_names
        # Filter for revolute joints (exclude fixed joints)
        kinematic_joints = []
        for joint_name in all_joints:
            #joint_prim_path = f"/World/Robot/{joint_name}"
            joint_prim_path = f"{self.prim_path}{joint_name}"

            joint_prim = get_prim_at_path(joint_prim_path)
            if joint_prim and joint_prim.HasAPI(UsdPhysics.RevoluteJointAPI):
                kinematic_joints.append(joint_name)
        
        return kinematic_joints
        '''



    def _map_joint_names(self, joint_names):
        """
        Map joint names from MuJoCo format to IsaacSim format
        """
        # Get actual joint names from the robot
        actual_joint_names = self.articulation.dof_names
        
        # If input names are in MuJoCo format (joint0, joint1, etc.)
        if all(name.startswith('joint') and name[5:].isdigit() for name in joint_names):
            # Map by index: joint0 -> first joint, joint1 -> second joint, etc.
            mapped_names = []
            for name in joint_names:
                joint_idx = int(name[5:])  # Extract number from "jointX"
                if joint_idx < len(actual_joint_names):
                    mapped_names.append(actual_joint_names[joint_idx])
                else:
                    raise Exception(f"Joint index {joint_idx} out of range. Robot has {len(actual_joint_names)} joints.")
            return mapped_names
        
        # If names are already in correct format, return as-is
        return joint_names

    def get_joint_positions(self):
        """Get current joint positions"""
        if hasattr(self, 'articulation'):
            all_positions = self.articulation.get_joint_positions()
            return all_positions[self.controlled_joint_indices]
        return None

    def get_joint_velocities(self):
        """Get current joint velocities"""
        if hasattr(self, 'articulation'):
            all_velocities = self.articulation.get_joint_velocities()
            return all_velocities[self.controlled_joint_indices]
        return None

    def set_joint_positions(self, positions):
        """Set joint positions"""
        if hasattr(self, 'articulation'):
            full_positions = self.articulation.get_joint_positions()
            for i, idx in enumerate(self.controlled_joint_indices):
                full_positions[idx] = positions[i]
            self.articulation.set_joint_positions(full_positions)

    def set_joint_velocities(self, velocities):
        """Set joint velocities"""
        if hasattr(self, 'articulation'):
            full_velocities = self.articulation.get_joint_velocities()
            for i, idx in enumerate(self.controlled_joint_indices):
                full_velocities[idx] = velocities[i]
            self.articulation.set_joint_velocities(full_velocities)





        

    def disconnect(self):
        """Any socket closing etc that must be done to properly shut down"""
        self.simulation_app.close() # close Isaac Sim
        print("IsaacSim connection closed...")


      
    def send_forces(self, u):
        """Applies the set of torques u to the arm."""
        
        # Create full torque vector for all DOFs
        total_dofs = self.articulation_view.num_dof
        full_torques = np.zeros(total_dofs)
        
        # Apply control torques to the controlled joints
        full_torques[:len(u)] = u
        #print("U: ", u)
        
        # Apply the control signal
        self.articulation_view.set_joint_efforts(full_torques)
        #self.articulation_view.set_joint_efforts(np.ones(total_dofs) * -10000)  # Set the joint efforts (torques)
        
        # Move simulation ahead one time step
        self.world.step(render=True)
        
    


    def send_target_angles(self, q):
        """Moves the arm to the specified joint angles

        q : numpy.array
                the target joint angles [radians]
        
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
        """
        self.set_joint_positions(q)  
        self.world.step(render=True)      
        
      


    
    def get_feedback(self, all_joints=False):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """
        self.q = self.get_joint_positions()
        self.dq = self.get_joint_velocities()

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
    