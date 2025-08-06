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
        self.joint_alt_pos_addrs = []
        self.joint_vel_addrs = []
        self.joint_dyn_addrs = []
        
        if joint_names is None:
            print("No joint names provided, using all controllable joints in the articulation.")
            joint_names = self.robot_config.controlled_joints

            


        # Validate joint names and get indices
        self.all_dof_names = self.articulation.dof_names
        # joint_names = self.articulation_view.joint_names  # The 24-joint list
        print(f"All dof names: {self.all_dof_names}")
        print(f"All link names: {self.articulation_view.body_names}")
        print(f"Provided joint names: {joint_names}")


        for name in joint_names:
            if name not in self.all_dof_names:
                raise Exception(f"Joint name {name} does not exist in robot model")
            #print(f"name: {name}")
            #link_name = name.replace("joint", "link")
            #print(f"link_name: {link_name}")
           

            #joint_idx = self.articulation_view.get_joint_index(name)
            dof_idx = self.articulation_view.get_dof_index(name)
            #print(f"dof_index: {dof_idx}")
            #print(f"joint_idx: {joint_idx}")
            # link_idx = self.articulation_view.get_link_index(link_name)
            # print(f"link_idx: {link_idx}")
            self.joint_pos_addrs.append(dof_idx)
            self.joint_vel_addrs.append(dof_idx)
            #TODO check if joint_dyn_addrs necessary
            self.joint_dyn_addrs.append(dof_idx)


        # Connect robot config with simulation data
        print("Connecting to robot config...")
        self.robot_config._connect(
            self.world,
            self.stage,
            self.articulation,
            self.articulation_view,
            self.joint_pos_addrs,
            self.joint_alt_pos_addrs,
            self.joint_vel_addrs,
            self.prim_path,
        )

        if self.robot_config.robot_type.startswith("h1"):
            self.world.add_physics_callback("send_actions", self.send_actions)

            

        



    def disconnect(self):
        """Any socket closing etc that must be done to properly shut down"""
        self.simulation_app.close() # close Isaac Sim
        print("IsaacSim connection closed...")

    '''
    def send_forces(self, u):
        """Applies the torques u to the joints specified in indices."""
        
        # Debug: Check array sizes and indices
        print(f"=== SEND_FORCES DEBUG ===")
        print(f"robot_config.N_ALL_JOINTS: {self.robot_config.N_ALL_JOINTS}")
        print(f"articulation_view joint count: {len(self.articulation_view.joint_names)}")
        print(f"joint_pos_addrs: {self.joint_pos_addrs}")
        print(f"u shape: {u.shape}")
        print(f"Max index in joint_pos_addrs: {max(self.joint_pos_addrs) if self.joint_pos_addrs else 'None'}")
        
        # Use the correct joint count for the articulation view
        total_joints = len(self.articulation_view.joint_names)  # Should be 24
        full_torques = np.zeros(total_joints)
        
        # Apply control torques to the controlled joints
        full_torques[self.joint_pos_addrs] = u
        
        print(f"full_torques shape: {full_torques.shape}")
        print(f"Non-zero torques at indices: {np.nonzero(full_torques)[0]}")
        
        # Apply the control signal
        self.articulation_view.set_joint_efforts(full_torques)
        
        # Move simulation ahead one time step
        self.world.step(render=True)
        '''



    def debug_dof_mapping(self):
        print("=== DOF MAPPING DEBUG ===")
        print(f"robot_config.N_ALL_JOINTS: {self.robot_config.N_ALL_JOINTS}")
        print(f"articulation.dof_names length: {len(self.articulation.dof_names)}")
        print(f"articulation_view.dof_names length: {len(self.articulation_view.dof_names)}")
        print(f"articulation_view.joint_names length: {len(self.articulation_view.joint_names)}")
        
        print("\nDOF names (actuated joints):")
        dof_names = self.articulation.dof_names
        for i, name in enumerate(dof_names):
            print(f"  {i}: {name}")
        
        print("\nLooking for controlled joints in DOF names:")
        for joint_name in self.robot_config.controlled_joints:
            if joint_name in dof_names:
                idx = dof_names.index(joint_name)
                print(f"  {joint_name}: DOF index {idx}")
            else:
                print(f"  {joint_name}: NOT FOUND in DOF names!")


    
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

        self.articulation_view.switch_control_mode(
            mode="effort",
            joint_indices=self.joint_pos_addrs
            )
        
        stiffness = np.ones(self.robot_config.N_ALL_JOINTS) * 100.0
        damping = np.ones(self.robot_config.N_ALL_JOINTS) * 10.0
        
        # Use the correct DOF indices
        for dof_idx in self.joint_pos_addrs:
            stiffness[dof_idx] = 20.0
            damping[dof_idx] = 5.0
            print(f"Set gains for DOF {dof_idx}: {self.all_dof_names[dof_idx]}")
        
        # Reshape for ArticulationView: (M, K)
        stiffness = np.expand_dims(stiffness, axis=0)
        damping = np.expand_dims(damping, axis=0)

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

