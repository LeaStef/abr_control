import numpy as np
import math
from isaacsim import SimulationApp
from .interface import Interface
simulation_app = SimulationApp({"headless": False}) 
import omni
import omni.kit.commands # type: ignore
import omni.isaac.core.utils.stage as stage_utils # type: ignore   
from omni.isaac.core import World # type: ignore
from omni.isaac.core.articulations import Articulation, ArticulationView # type: ignore
#TODO change import 
#TODO is "Robot" even necessary 
from isaacsim.core.api.robots import Robot # type: ignore
from pxr import UsdGeom, Gf, UsdShade, Sdf, UsdPhysics# type: ignore  
from omni.isaac.core.utils.nucleus import get_assets_root_path # type: ignore
from isaacsim.robot.policy.examples.robots import H1FlatTerrainPolicy # type: ignore
import omni.isaac.core.utils.numpy.rotations as rot_utils  # type: ignore


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
        #self.count = 0  # keep track of how many times send forces is called
        self.prim_path = "/World/robot"
        #remove?
        self.name = self.robot_config.robot_type 
        

    def connect(self, joint_names=None, camera_id=-1):
        """
        joint_names: list, optional (Default: None)
            list of joint names to send control signal to and get feedback from
            if None, the joints in the kinematic tree connecting the end-effector
            to the world are used
        """

        """All initial setup."""
        # Initialize the simulation world
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.context = omni.usd.get_context()
        self.stage = self.context.get_stage()


        # Load the robot from USD file
        assets_root_path = get_assets_root_path()
        robot_usd_path = f"{assets_root_path}{self.robot_config.robot_path}"
        print(f"Robot '{self.robot_config.robot_type}' is loaded from USD path: {robot_usd_path}")
        
        if self.robot_config.robot_type.startswith("h1"):
            self.h1 = H1FlatTerrainPolicy(
                prim_path=self.prim_path,
                name=self.name,
                usd_path=robot_usd_path,
                position=np.array([0, 0 , 0]),
                orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
            )

        else:    
            stage_utils.add_reference_to_stage(
                    usd_path=robot_usd_path,
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

        
        
        
        # add virtual EE if none exists
        if (self.robot_config.has_EE is False):
            print("Robot has no EE, virtual one is attached.")
            self.add_virtual_ee_link(self.robot_config.EE_parent_link, self.robot_config.ee_link_name)
        

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
        print(all_joint_names)
        for name in joint_names:
            if name not in all_joint_names:
                raise Exception(f"Joint name {name} does not exist in robot model")
        
            joint_idx = all_joint_names.index(name)
            self.joint_pos_addrs.append(joint_idx)
            self.joint_vel_addrs.append(joint_idx)
            #TODO check if joint_dyn_addrs necessary
            self.joint_dyn_addrs.append(joint_idx)


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
        )

        if self.robot_config.robot_type.startswith("h1"):
            self.world.add_physics_callback("send_actions", self.send_actions)

            




        


    def _map_joint_names(self, joint_names):
        """
        Map joint names from MuJoCo format to IsaacSim format
        """
        # Get actual joint names from the robot
        actual_joint_names = self.articulation.dof_names

        if self.name is "h1":
            joint_list = [#'torso_joint',                    
                          'right_shoulder_pitch_joint',     
                          'right_shoulder_roll_joint',      
                          'right_shoulder_yaw_joint',       
                          'right_elbow_joint'] ,
                          #'right_wrist_joint']              
            return np.array(joint_list)
        
        elif self.name is "h1_hands":
            joint_list = ['right_shoulder_pitch_joint',     
                          'right_shoulder_roll_joint',      
                          'right_shoulder_yaw_joint',       
                          'right_elbow_joint',
                          'right_hand_joint']              
            return np.array(joint_list)
        else:
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
        """Applies the torques u to the joints specified in indices."""
        #print ("send forces")
        # Create full torque vector for all DOFs
        full_torques = np.zeros(self.robot_config.N_ALL_JOINTS)
        # Apply control torques to the controlled joints
        full_torques[self.joint_pos_addrs] = u
        # Apply the control signal
        #TODO maybe outsource as done for position
        self.articulation_view.set_joint_efforts(full_torques)
        # Move simulation ahead one time step
        self.world.step(render=True)
    '''


    def send_forces(self, u):
        """Applies the torques u to the joints specified in indices."""
        
        #def test_single_joint_force(self, joint_index, force_value=2.0):
        

        joint_list = ['right_shoulder_pitch_joint',     
                          'right_shoulder_roll_joint',      
                          'right_shoulder_yaw_joint',       
                          'right_elbow_joint',
                          'right_hand_joint']  

        # Create zero torque array
        test_torques = np.zeros(self.robot_config.N_ALL_JOINTS)

        test_torques[self.joint_pos_addrs] = u * 0.1
        
    
        print(f"Applied torque: {u} ")
        print(f"Torque array: {test_torques}")
        
        # Apply the torque
        self.articulation_view.set_joint_efforts(test_torques)
        
        # Let it run for a moment to observe
        for _ in range(100):  # Run for ~1 second at 100Hz
            self.world.step(render=True)
        
        print("Observe which joint moved and in what direction")
        print("Press Enter to continue to next joint...")
        input()
         '''


       

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
        _cube =  self.stage.GetPrimAtPath(prim_path)
        # Check if it's an Xformable
        if not _cube.IsValid() or not UsdGeom.Xformable(_cube):
            print(f"Prim at {_cube.GetPath()} is not a valid Xformable.")
        else:
            xformable = UsdGeom.Xformable(_cube)
        # Get the local transformation matrix
        transform_matrix = xformable.GetLocalTransformation()

        return transform_matrix


    def set_xyz(self, prim_path, xyz, orientation=np.array([0., 0., 0., 1.])):
        """Set the position of an object in the environment.

        prim_path : string
            the prim_path of the object
        xyz : np.array
            the [x,y,z] location of the target [meters]
        """     
        _cube =  self.stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(_cube)
        transform_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(xyz[0], xyz[1], xyz[2]))
        xformable.MakeMatrixXform().Set(transform_matrix)


    # method for keep_standing
    def send_actions(self, dt):
        pelvis_prim_path = '/World/robot/pelvis'  
        prim = self.stage.GetPrimAtPath(pelvis_prim_path)
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.0, 0.0, 1.4))
        #prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0.70711 ,0.70711 ,0.0 ,0.0))

    
   

    #TODO check if position is even set here. maybe remove
    # Create a visual-only cube (no collision)
    def create_target_prim(self, prim_path="/World/target_cube", position=np.array([0, 0, 1.0]), size = .1, color=np.array([0, 0, 1.0])):        
        # Create cube geometry
        cube_prim = UsdGeom.Cube.Define(self.stage, prim_path)
        cube_prim.CreateSizeAttr(size)  # Unit cube
        
        # Set transform (position and scale)
        xformable = UsdGeom.Xformable(cube_prim)
        transform_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(position[0], position[1], position[2]))
        xformable.MakeMatrixXform().Set(transform_matrix)
        # xformable.AddTranslateOp().Set(Gf.Vec3f(*position))

        
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

    

    def set_gains_force_control(self):
        """Properly set gains for arm joints (DOFs 0-5) and finger joints if present"""
        
        # Get current gains or set defaults
        stiffness = np.ones(self.robot_config.N_ALL_JOINTS) * 100.0  # Default high stiffness
        damping = np.ones(self.robot_config.N_ALL_JOINTS) * 10.0     # Default damping
        
        # Set controlled arm joints to zero stiffness for force control
        for idx in self.joint_pos_addrs:
            stiffness[idx] = 0.0    # Zero stiffness = force control
            damping[idx] = 0.1      # Low damping for responsiveness
        
  
        self.articulation_view.set_gains(stiffness, damping)
        print(f"Set gains for force control for arm joints {self.joint_pos_addrs}")
    

    
    def set_gains_force_control_h1(self):
        stiffness = np.ones(self.robot_config.N_ALL_JOINTS) * 100.0
        damping = np.ones(self.robot_config.N_ALL_JOINTS) * 10.0
        
        for idx in self.joint_pos_addrs:
            stiffness[idx]  =  20.0 # 4. # Small but non-zero stiffness
            damping[idx]    =  5.0   # 1.0   # Higher damping for stability

        self.articulation_view.set_gains(stiffness, damping)



    def add_virtual_ee_link(self, EE_parent_link, ee_name, offset=[-0.04, 0, 0]):
        """Add virtual end effector link as an Xform under the specified parent link"""
        # Full path to parent
        parent_path = f"{self.prim_path}/{EE_parent_link}"

        # Full path to the new EE transform, nested under parent
        ee_prim_path = f"{parent_path}/{ee_name}"

        # Create the Xform prim
        ee_prim = UsdGeom.Xform.Define(self.stage, ee_prim_path)

        # Set transform relative to parent
        ee_prim.AddTranslateOp().Set(Gf.Vec3d(*offset))

        print(f"Created virtual EE link at {ee_prim_path}")
