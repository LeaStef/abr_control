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
        self.prim_path = "/World/robot"
        self.name = self.robot_config.xml_dir.rsplit('/', 1)[-1]
        

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
        import omni.kit.commands # type: ignore
        import omni.isaac.core.utils.stage as stage_utils # type: ignore   
        from omni.isaac.core import World # type: ignore
        from omni.isaac.core.articulations import Articulation, ArticulationView # type: ignore
        from omni.isaac.core.utils.nucleus import get_assets_root_path # type: ignore
        #TODO change import 
        #TODO is "Robot" even necessary 
        from isaacsim.core.api.robots import Robot # type: ignore
        
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
            #ee_path = self.create_ee_xform("wrist_3_link", np.array([-0.04, 0, 0]))
            has_EE = False  # UR5 has no end-effector
            EE_parent_link = "wrist_3_link"  # end-effector link for UR5
            self.ee_link_name = "flange"  # end-effector name for UR5
        elif self.robot_config.robot == "jaco2":
            robot_path = "/Isaac/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd"
            has_EE = True  # Jaco2 has an end-effector
            self.ee_link_name = "j2n6s300_end_effector"  # end-effector name for Jaco2
        elif self.robot_config.robot == "h1":   
            robot_path = "/Isaac/Robots/Unitree/H1/h1.usd"
            #TODO check this
            EE_parent_link = "link6"  # end-effector link for H1
        print(f"Robot '{self.name}' is loaded from UDS ...")
        print(f"End Effector with name '{self.ee_link_name}' specified in UDS, using it ...")

        assets_root_path = get_assets_root_path()
        robot_usd_path = f"{assets_root_path}{robot_path}"
        print(f"Loading robot from USD path: {robot_usd_path}")
        
        
        stage_utils.add_reference_to_stage(
                 usd_path=assets_root_path + robot_path,
                 prim_path=self.prim_path,
                 )
        robot = self.world.scene.add(Robot(prim_path=self.prim_path, name=self.name))
        
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
        else:
            # Handle joint name mapping
            joint_names = self._map_joint_names(joint_names)

        # Validate joint names and get indices
        all_joint_names = self.articulation.dof_names
        
        for name in joint_names:
            if name not in all_joint_names:
                raise Exception(f"Joint name {name} does not exist in robot model")
            #TODO check if  joint_idx is necessary
            joint_idx = all_joint_names.index(name)
            self.joint_pos_addrs.append(joint_idx)
            self.joint_vel_addrs.append(joint_idx)
            self.joint_dyn_addrs.append(joint_idx)
        

        '''
        # is boolean flag necessary?
        if has_EE:
            print(f"End Effector with name '{self.ee_link_name}' specified in UDS, using it ...")
        else:
            print("No End Effector, creating EE Xform...")
            ee_path = self.create_ee_xform(EE_parent_link, np.array([-0.04, 0, 0]))
        #print("End Effector Info: ", ee_info)
        '''


        # Connect robot config with simulation data
        print("Connecting to robot config...")
        print("Joint Position Addresses: ", self.joint_pos_addrs)
        print("Joint Velocity Addresses: ", self.joint_vel_addrs)
        self.robot_config._connect(
            self.world,
            self.stage,
            self.articulation,
            self.articulation_view,
            self.joint_pos_addrs,
            self.joint_vel_addrs,
            self.prim_path,
            self.ee_link_name
        )



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


    def disconnect(self):
        """Any socket closing etc that must be done to properly shut down"""
        self.simulation_app.close() # close Isaac Sim
        print("IsaacSim connection closed...")


    def send_forces(self, u):
        """Applies the set of torques u to the arm - now working correctly!"""
        # Create full torque vector for all DOFs
        full_torques = np.zeros(self.robot_config.N_ALL_JOINTS)
        # Apply control torques to the controlled joints
        full_torques[:len(u)] = u
        # Apply the control signal
        self.articulation_view.set_joint_efforts(full_torques)
        # Move simulation ahead one time step
        self.world.step(render=True)


    def send_target_angles(self, q):
        """Moves the arm to the specified joint angles

        q : numpy.array
                the target joint angles [radians]
        """
        self.robot_config.set_joint_positions(q)
        self.world.step(render=True)


    def get_feedback(self):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """
        self.q = self.robot_config.get_joint_positions()
        self.dq = self.robot_config.get_joint_velocities()
        return {"q": self.q, "dq": self.dq}


    def get_xyz(self, prim_path):
                """Returns the xyz position of the specified object

                prim_path : string
                    path of the object you want the xyz position of
                """
                transform_matrix = self.get_transform(prim_path)
                translation = transform_matrix.ExtractTranslation()
                return np.array([translation[0], translation[1], translation[2]], dtype=np.float64)


    #TODO check if overlap to def quaternion
    def get_orientation(self, prim_path):
        """Returns the orientation of an object in IsaacSim
        Parameters
        ----------
        name : string
            the name of the object of interest
        """
        transform_matrix = self.get_transform(prim_path)
        quat = transform_matrix.ExtractRotationQuat()
        quat_np = np.array([quat.GetReal(), *quat.GetImaginary()])
        return quat_np
    

    def get_transform(self, prim_path):
        from pxr import Usd, UsdGeom, Gf
        _cube =  self.stage.GetPrimAtPath(prim_path)
        # Check if it's an Xformable
        if not _cube.IsValid() or not UsdGeom.Xformable(_cube):
            print(f"Prim at {_cube.GetPath()} is not a valid Xformable.")
        else:
            xformable = UsdGeom.Xformable(_cube)
        # Get the local transformation matrix
        transform_matrix = xformable.GetLocalTransformation()

        return transform_matrix



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
    def set_xyz(self, prim_path, xyz, orientation=np.array([0., 0., 0., 1.])):
        """Set the position of an object in the environment.

        name : string
            the name of the object
        xyz : np.array
            the [x,y,z] location of the target [meters]
        """
        from pxr import UsdGeom, Gf, UsdShade, Sdf, UsdPhysics # type: ignore        

        _cube =  self.stage.GetPrimAtPath(prim_path)
        #_cube.set_world_pose(xyz, orientation) # set the position and orientation of the object

        xformable = UsdGeom.Xformable(_cube)
        transform_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(xyz[0], xyz[1], xyz[2]))
        xformable.MakeMatrixXform().Set(transform_matrix)


    # method for keep_standing
    def send_actions(self, dt):
        pelvis_prim_path = '/World/robot/pelvis'
        from pxr import Gf  # type: ignore    
        prim=self.stage.GetPrimAtPath(pelvis_prim_path)
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0 ,0.0 ,0.0 ,0.0))
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.0 ,0.0 ,0.02))
        #prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0.70711 ,0.70711 ,0.0 ,0.0))



    # Create a visual-only cube (no collision)
    def create_target_prim(self, prim_path="/World/target_cube", position=np.array([0, 0, 1.0]), size = .1, color=np.array([0, 0, 1.0])):
        from pxr import UsdGeom, Gf, UsdShade, Sdf, UsdPhysics # type: ignore        
        
        # Create cube geometry
        cube_prim = UsdGeom.Cube.Define(self.stage, prim_path)
        cube_prim.CreateSizeAttr(size)  # Unit cube
        
        # Set transform (position and scale)
        xformable = UsdGeom.Xformable(cube_prim)
        transform_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(position[0], position[1], position[2]))
        xformable.MakeMatrixXform().Set(transform_matrix)
        
        # Create and apply material for color
        material_path = prim_path + "/Material"
        material = UsdShade.Material.Define(self.stage, material_path)
        
        # Create shader
        shader = UsdShade.Shader.Define(self.stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[0], color[1], color[2]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        
        # Connect shader to material
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        # Bind material to cube
        UsdShade.MaterialBindingAPI(cube_prim).Bind(material)
        
        # Disable collision to ensure it's purely visual
        cube_prim.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

        return cube_prim
    

    #TODO not sure why this is necessary
    def fix_arm_joint_gains(self):
        """Properly set gains for arm joints (DOFs 0-5)"""
        n_dofs = self.robot_config.N_ALL_JOINTS
        
        # Get current gains or set defaults
        stiffness = np.ones(n_dofs) * 100.0  # Default high stiffness
        damping = np.ones(n_dofs) * 10.0     # Default damping
        
        # Set ARM joints (0-5) to zero stiffness for force control
        arm_joint_indices = [0, 1, 2, 3, 4, 5]  # These are your arm joints
        
        for idx in arm_joint_indices:
            stiffness[idx] = 0.0    # Zero stiffness = force control
            damping[idx] = 0.1      # Low damping for responsiveness
        
        # Keep finger joints with some stiffness if you want them stable
        finger_joint_indices = [6, 7, 8, 9, 10, 11]
        for idx in finger_joint_indices:
            stiffness[idx] = 50.0   # Moderate stiffness for fingers
            damping[idx] = 5.0      # Moderate damping for fingers
        
        self.articulation_view.set_gains(stiffness, damping)
        print("Fixed gains for arm joints (0-5)")
