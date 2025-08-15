import numpy as np
from isaacsim import SimulationApp
from .interface import Interface
simulation_app = SimulationApp({"headless": False}) 
import omni
import omni.kit.commands # type: ignore
import isaacsim.core.utils.stage as stage_utils # type: ignore   
from omni.isaac.core import World # type: ignore
from omni.isaac.core.articulations import ArticulationView # type: ignore
#from isaacsim.core.prims import Articulation # type: ignore
from isaacsim.core.api.robots import Robot # type: ignore
from pxr import UsdGeom, Gf, UsdShade, Sdf # type: ignore
from isaacsim.core.utils.nucleus import get_assets_root_path # type: ignore
import isaacsim.core.utils.numpy.rotations as rot_utils  # type: ignore


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
        self.robot_config = robot_config
        self.dt = dt  # time step
        self.prim_path = "/World/robot"


    def connect(self, joint_names=None):
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
        
        stage_utils.add_reference_to_stage(
            usd_path=robot_usd_path,
            prim_path=self.prim_path,
            )
        robot = self.world.scene.add(Robot(prim_path=self.prim_path, name=self.robot_config.robot_type))
      
        self.world.reset()
        
        self.articulation_view = ArticulationView(prim_paths_expr=self.prim_path, name=self.robot_config.robot_type + "_view")
        self.world.scene.add(self.articulation_view)
        self.articulation_view.initialize()
        
        # add virtual EE if none exists
        if (self.robot_config.has_EE is False):
            print("Robot has no EE, virtual one is attached.")
            self.add_virtual_ee_link(self.robot_config.EE_parent_link, self.robot_config.ee_link_name, offset=self.robot_config.ee_offset)

        # Set simulation time step
        self.world.get_physics_context().set_physics_dt(self.dt)
        
        # Reset the world to initialize physics
        self.world.reset()
        
        # Get joint information
        self.dof_indices = []
        self.joint_indices = []
        
        if joint_names is None:
            print("No joint names provided, using all controllable joints.")
            joint_names = self.robot_config.controlled_dof

        self.all_dof_names = self.articulation_view.dof_names
        self.all_joint_names = self.articulation_view.joint_names
        self.all_body_names = self.articulation_view.body_names 

        for name in joint_names:
            if name not in self.all_dof_names and name not in self.all_joint_names:
                raise Exception(f"Joint name {name} does not exist in robot model")
            joint_idx = self.articulation_view.get_joint_index(name)
            dof_idx = self.articulation_view.get_dof_index(name)
            self.dof_indices.append(dof_idx)
            self.joint_indices.append(joint_idx)
           

        # Connect robot config with simulation data
        print("Connecting to robot config...")
        self.robot_config._connect(
            self.world,
            self.stage,
            self.articulation_view,
            self.dof_indices,
            self.joint_indices,
            self.prim_path,
        )

        if self.robot_config.robot_type.startswith("h1"):
            self.world.add_physics_callback("keep_standing", self.keep_standing)


    def disconnect(self):
        """Any socket closing etc that must be done to properly shut down"""
        self.simulation_app.close() # close Isaac Sim
        print("IsaacSim connection closed...")

    
    def send_forces(self, u):
        """Applies the torques u to the joints specified in indices."""
        # Create full torque vector for all DOFs
        full_torques = np.zeros(self.robot_config.N_ALL_DOF)
        # Apply control torques to the controlled joints
        full_torques[self.dof_indices] = u 
        # Apply the control signal
        self.articulation_view.set_joint_efforts(full_torques)
        # Move simulation ahead one time step
        self.world.step(render=True)
    

    def send_target_angles(self, q):
        """Moves the arm to the specified joint angles

        q : numpy.array
                the target joint angles [radians]
        """
        self.robot_config._set_joint_positions(q)
        self.world.step(render=True)


    def get_feedback(self):
        """Return a dictionary of information needed by the controller.
        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """
        self.q = self.robot_config._get_joint_positions()
        self.dq = self.robot_config._get_joint_velocities()
        return {"q": self.q, "dq": self.dq}
    
    
    def set_xyz(self, prim_path, xyz):
        """Set the position of an object in the environment.

        prim_path : string
            the prim_path of the object
        xyz : np.array
            the [x,y,z] location of the target [meters]
        """     
        prim =  self.stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        transform_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(xyz[0], xyz[1], xyz[2]))
        xformable.MakeMatrixXform().Set(transform_matrix)
    

    # method that humanoid does not fall over
    def keep_standing(self, dt):
        prim = self.stage.GetPrimAtPath(self.robot_config.lock_prim_standing)
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.0, 0.0, 1.4))
   

    # Create a visual-only cube (no collision)
    def create_target_prim(self, prim_path="/World/target", size = .1, color=np.array([0, 0, 1.0])):        
        # Create cube geometry
        cube_prim = UsdGeom.Cube.Define(self.stage, prim_path)
        cube_prim.CreateSizeAttr(size)  
        
        # Create and apply material for color
        material_path = prim_path + "/Material"
        material = UsdShade.Material.Define(self.stage, material_path)
        
        # Create shader
        shader = UsdShade.Shader.Define(self.stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color[0], color[1], color[2]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        
        # Connect shader to material and bind material to cube
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube_prim).Bind(material)

        # Disable collision to ensure it's purely visual
        cube_prim.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

        return cube_prim

    
    def set_gains_force_control(self):
        """Properly set gains for arm joints (DOFs 0-5) and finger joints if present"""
        # Get current gains or set defaults
        stiffness = np.ones(self.robot_config.N_ALL_DOF) * 100.0  # Default high stiffness
        damping = np.ones(self.robot_config.N_ALL_DOF) * 10.0     # Default damping
        
        if self.robot_config._is_fixed_base():
            controlled_s = 0.0
            controlled_d = 0.1
        else:
            controlled_s = 2.0
            controlled_d = 0.5

        for idx in self.dof_indices:
            stiffness[idx] = controlled_s
            damping[idx] = controlled_d
        
        self.articulation_view.set_gains(stiffness, damping)
        print(f"Set gains for force control for arm joints {self.dof_indices}")
    

    def add_virtual_ee_link(self, EE_parent_link, ee_name, offset):
        """Add virtual end effector link as an Xform under the specified parent link"""
        parent_path = f"{self.prim_path}/{EE_parent_link}"
        # Full path to the new EE transform, nested under parent
        ee_prim_path = f"{parent_path}/{ee_name}"
        # Create the Xform prim
        ee_prim = UsdGeom.Xform.Define(self.stage, ee_prim_path)
        # Set transform relative to parent
        ee_prim.AddTranslateOp().Set(Gf.Vec3d(*offset))
