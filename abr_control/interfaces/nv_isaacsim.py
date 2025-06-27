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
        import omni.kit.commands  # type: ignore
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
            #ee_path = self.create_ee_xform("wrist_3_link", np.array([-0.04, 0, 0]))
            EE_parent_link = "wrist_3_link"  # end-effector link for UR5
            has_EE = False  # UR5 has no end-effector
            self.ee_link_name = "flange" #"ft_frame" # end-effector name for UR5
        elif self.robot_config.robot == "jaco2":
            robot_path = "/Isaac/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd"
            EE_link = "j2n6s300_end_effector"
            EE_parent_link = "j2n6s300_link_6"  # end-effector link for Jaco2
            has_EE = True  # Jaco2 has an end-effector
            #omni.kit.commands.execute("MovePrim", path_from="/World/robot/j2n6s300_end_effector", path_to="/World/robot/EE")
            #ee_path = self.create_ee_xform("j2n6s300_link_6", np.array([-0.04, 0, 0]))
            self.ee_link_name = "end_effector"  # end-effector name for Jaco2
        elif self.robot_config.robot == "h1":   
            robot_path = "/Isaac/Robots/Unitree/H1/h1.usd"
            #TODO check this
            EE_parent_link = "link6"  # end-effector link for H1


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
        else:
            # Handle joint name mapping
            joint_names = self._map_joint_names(joint_names)



        # Validate joint names and get indices
        all_joint_names = self.articulation.dof_names
        
        for name in joint_names:
            if name not in all_joint_names:
                raise Exception(f"Joint name {name} does not exist in robot model")
            joint_idx = all_joint_names.index(name)
            self.joint_pos_addrs.append(joint_idx)
            self.joint_vel_addrs.append(joint_idx)
            self.joint_dyn_addrs.append(joint_idx)
        
        # is boolean flag necessary?
        if has_EE:
            print("End Effector ...")
            # 'EE_link = "j2n6s300_end_effector"'
        else:
            print("No End Effector, creating EE Xform...")
            #ee_info = self.initialize_robot_end_effector(self.prim_path, EE_parent_link, np.array([-0.04, 0, 0]))
            #ee_path = self.create_ee_xform(EE_parent_link, np.array([-0.04, 0, 0]))
        #print("End Effector Info: ", ee_info)
        


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
        )





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

        #full_torques[:len(u)] = np.ones(len(u)) * -1000
        #full_torques[1] =  -1000
        #self.articulation_view.set_joint_efforts(full_torques)  # Set the joint efforts (torques)

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
    

    #TODO remove
    def find_EE(self, EE_name="end_effector"):
        """Find the end-effector"""
        
        link_list = self.articulation_view.body_names


        
        # Check if the articulation has an end-effector link
        ee_link_name = None
        for link_name in self.articulation_view.body_names:
            if "end_effector" in link_name.lower() or "ee" in link_name.lower():
                ee_link_name = link_name
                break
        
        if ee_link_name is None:
            raise Exception("End Effector link not found in the articulation.")
        
        # Return the name of the end-effector link
        return ee_link_name


    # Create an invisible Xform for EE (wrist_3_link for UR5)
    def create_ee_xform(self, parent_link="link6", ee_offset=np.array([-0.04, 0, 0])):
        """Create an invisible Xform node for End Effector queries
        
        Parameters
        ----------
        parent_link : str
            Name of the final link
        ee_offset : np.array
            Offset from parent_link to EE point [x, y, z]
        """
        from pxr import UsdGeom, Gf, UsdPhysics, Sdf
        
        # Create an Xform (transform node) for the EE
        ee_path = f"{self.prim_path}/{parent_link}/EE"
        print("ee path: ", ee_path)
        ee_xform = UsdGeom.Xform.Define(self.stage, ee_path)

        # Set position relative to parent_link 
        ee_xform.AddTranslateOp().Set(Gf.Vec3d(ee_offset[0], ee_offset[1], ee_offset[2]))

        # Add minimal physics properties without creating a separate rigid body
        # This makes it part of the parent link's rigid body
        ee_xform.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    
        # Add a custom attribute to mark it as an end effector for easier identification
        ee_xform.GetPrim().CreateAttribute("custom:isEndEffector", Sdf.ValueTypeNames.Bool).Set(True)
    
        return ee_path


    def get_ee_link_info(self, parent_link="link6"):
        """Get information about all links including the EE
        
        Returns
        -------
        dict
            Dictionary with link names as keys and indices as values
        """
        if hasattr(self, 'articulation_view'):
            # Get all existing link names
            link_names = self.articulation_view.body_names
            
            # Check if EE exists as a child of link6
            ee_path = f"{self.prim_path}/{parent_link}/EE"
            ee_prim = self.stage.GetPrimAtPath(ee_path)
            
            if ee_prim.IsValid():
                # Add EE to the link names list
                link_names.append("EE")
                
            # Create a mapping of names to indices
            link_info = {name: idx for idx, name in enumerate(link_names)}
            return link_info
        
        return {}
