from xml.etree import ElementTree
import numpy as np
from abr_control.utils import download_meshes
import omni






class IsaacsimConfig:
    """A wrapper on the Mujoco simulator to generate all the kinematics and
    dynamics calculations necessary for controllers.
    """

    # https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/Joints.html
    JNT_POS_LENGTH_ISAACSIM = {
        "free": 7,   # 3 (translation) + 4 (quaternion rotation), usually not used in articulated chains
        "spherical": 4,  # Represented as quaternion in PxArticulationReducedCoordinate
        "prismatic": 1,  # Linear motion in one axis
        "revolute": 1,   # Rotational motion in one axis
    }
    '''
    from omni.isaac.core.articulations import ArticulationJointType
    JNT_POS_LENGTH = {
        ArticulationJointType.FREE: 7,   # 3 for position + 4 for quaternion orientation
        ArticulationJointType.BALL: 4,   # quaternion orientation
        ArticulationJointType.PRISMATIC: 1,  # 1 DoF linear
        ArticulationJointType.REVOLUTE: 1,   # 1 DoF rotational
    }
    '''

    JNT_DYN_LENGTH_ISAACSIM = {
        "free": 6,        # 3 linear + 3 angular velocity (used for root link only, not in articulated chains)
        "spherical": 3,   # Angular velocity vector (3D)
        "prismatic": 1,   # Linear velocity along one axis
        "revolute": 1,    # Angular velocity around one axis
    }
    



    def __init__(self, robot_type, folder=None, use_sim_state=True, force_download=False):
        """Loads the Isaacsim model from the specified xml file

        Parameters
        ----------
        xml_file: string
            the name of the arm model to load. If folder remains as None,
            the string passed in is parsed such that everything up to the first
            underscore is used for the arm directory, and the full string is
            used to load the xml within that folder.

            EX: 'myArm' and 'myArm_with_gripper' will both look in the
            'myArm' directory, however they will load myArm.xml and
            myArm_with_gripper.xml, respectively

            If a folder is passed in, then folder/xml_file is used
        folder: string, Optional (Default: None)
            specifies what folder to find the xml_file, if None specified will
            checking in abr_control/arms/xml_file (see above for xml_file)
        use_sim_state: Boolean, optional (Default: True)
            If set False, the state information is provided by the user, which
            is then used to calculate the corresponding dynamics values.
            The state is then set back to the sim state prior to the user
            provided state.
            If set true, any q and dq values passed in to the functions are
            ignored, and the current state of the simulator is used to
            calculate all functions. This can speed up the simulation, because
            the step of resetting the state on every call is omitted.
        force_download: boolean, Optional (Default: False)
            True to force downloading the mesh and texture files, useful when new files
            are added that may be missing.
            False: if the meshes folder is missing it will ask the user whether they
            want to download them
        """
        self.robot_type = robot_type
        #TODO obsolete, from mujoco
        self.use_sim_state = use_sim_state
         
        self.robot_path = None
        self.ee_link_name = "EE"  # name used for virtual end-effector, overwritten if one exists already
        
        if self.robot_type == "ur5":
            self.robot_path = "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
            self.has_EE = False  # UR5 has no end-effector
            self.EE_parent_link = "wrist_3_link"  
            START_ANGLES = "0 -.67 -.67 0 0 0"
            print(f"Virtual end effector with name '{self.ee_link_name}' is attached as robot has none.")

        elif self.robot_type == "jaco2":
            self.robot_path = "/Isaac/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd"
            self.has_EE = True  # jaco2 has an end-effector
            self.ee_link_name = "j2n6s300_end_effector"  
            START_ANGLES = "2.0 3.14 1.57 4.71 0.0 3.04"
            print(f"End effector with name '{self.ee_link_name}' specified in UDS, using it ...")
                
        elif self.robot_type == "h1":   
            self.robot_path = "/Isaac/Robots/Unitree/H1/h1.usd"
            self.has_EE = False  # H1 has no end-effector
            self.EE_parent_link = "right_elbow_link"
            START_ANGLES = "0 0 0 0"
            print(f"Virtual end effector with name '{self.ee_link_name}' is attached as robot has none.")

        self.START_ANGLES = np.array(START_ANGLES.split(), dtype=float)

        

    def _connect(self, world, stage, articulation, articulation_view, joint_pos_addrs, joint_vel_addrs, prim_path):

        """Called by the interface once the Mujoco simulation is created,
        this connects the config to the simulator so it can access the
        kinematics and dynamics information calculated by Mujoco.

        Parameters
        ----------
        sim: MjSim
            The Mujoco Simulator object created by the Mujoco Interface class
        joint_pos_addrs: np.array of ints
            The index of the robot joints in the Mujoco simulation data joint
            position array
        joint_vel_addrs: np.array of ints
            The index of the robot joints in the Mujoco simulation data joint
            Jacobian, inertia matrix, and gravity vector
        """

        # get access to the Isaac simulation
        self.world = world
        self.stage = stage
        self.articulation = articulation
        self.articulation_view = articulation_view
        self.joint_pos_addrs = np.copy(joint_pos_addrs)
        self.joint_vel_addrs = np.copy(joint_vel_addrs)
        self.prim_path = prim_path
        self.N_JOINTS = len(self.joint_vel_addrs)
        # number of joints in the IsaacSim simulation
        N_ALL_JOINTS = self.articulation_view.num_dof

        # need to calculate the joint_vel_addrs indices in flat vectors returned
        # for the Jacobian
        self.jac_indices = np.hstack(
            # 6 because position and rotation Jacobians are 3 x N_JOINTS
            [self.joint_vel_addrs + (ii * N_ALL_JOINTS) for ii in range(3)]
        )

        # for the inertia matrix
        self.M_indices = [
            ii * N_ALL_JOINTS + jj
            for jj in self.joint_vel_addrs
            for ii in self.joint_vel_addrs
        ]

        # a place to store data returned from Mujoco
        self._g = np.zeros(self.N_JOINTS)
        self._J3NP = np.zeros((3, N_ALL_JOINTS))
        self._J3NR = np.zeros((3, N_ALL_JOINTS))
        self._J6N = np.zeros((6, self.N_JOINTS))
        self._MNN = np.zeros((N_ALL_JOINTS, N_ALL_JOINTS))
        self._R9 = np.zeros(9)
        self._R = np.zeros((3, 3))
        self._x = np.ones(4)
        self.N_ALL_JOINTS = N_ALL_JOINTS




    def g(self, q=None):
        """
        Returns the gravity and Coriolis/centrifugal forces for the controlled arm_joints.
        Args:
            q (np.ndarray, optional): Joint positions
        Returns:
            np.ndarray: Generalized bias forces for controlled DOF
        """
        # Compute gravity and Coriolis/centrifugal separately
        gravity = self.articulation_view.get_generalized_gravity_forces(joint_indices=self.joint_pos_addrs)[0]
        coriolis = self.articulation_view.get_coriolis_and_centrifugal_forces(joint_indices=self.joint_pos_addrs)[0]
        
        g = gravity + coriolis

        return -g  
    
        

    def dJ(self, name, q=None, dq=None, x=None):
        """Returns the derivative of the Jacobian wrt to time

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        dq: float numpy.array, optional (Default: None)
            The joint velocities of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        # Note from Emo in Mujoco forums:
        # 'You would have to use a finate-difference approximation in the
        # general case, check differences.cpp'
        raise NotImplementedError


    
    def J(self, name, q=None, x=None, object_type="body"):
        if name == "EE": 
            name = self.ee_link_name
        
        # Check for unsupported features
        if x is not None and not np.allclose(x, 0):
            raise Exception("x offset currently not supported, set to None")
        
        if object_type == "body":
            # Get Jacobians from articulation view - ensure proper tensor handling
            jacobians = self.articulation_view.get_jacobians(clone=True)
            
            # Convert to numpy if it's a tensor
            if hasattr(jacobians, 'cpu'):
                jacobians = jacobians.cpu().numpy()
            elif hasattr(jacobians, 'numpy'):
                jacobians = jacobians.numpy()
            
            # Check if ArticulationView has any environments
            if jacobians.shape[0] == 0:
                raise RuntimeError("ArticulationView contains no environments. Make sure it's properly initialized and contains articulations.")
            
            # jaco2 version
            #link_index = self.articulation_view.get_link_index(name)
            #link_index = self.N_JOINTS -1
            #print("link_index old: ", link_index)
            #link_index = self.articulation_view.get_link_index(name)
            #print("name: ", name, "     link_index new: ", link_index)

            if self.robot_type is "ur5":
                link_index = 5

            elif self.robot_type is "jaco2":
                link_index = 6

            
            # Extract Jacobian for specific link
            env_idx = 0  # Assuming single environment
            
            # shape is (1, 13, 6, 12)
            if len(jacobians.shape) == 4:
                # Shape: (num_envs, num_bodies, 6, num_dofs)
                J_full = jacobians[env_idx, link_index, :, :]
            elif len(jacobians.shape) == 3:
                # Shape: (num_envs, num_bodies * 6, num_dofs)
                start_row = link_index * 6
                end_row = start_row + 6
                J_full = jacobians[env_idx, start_row:end_row, :]
            else:
                raise RuntimeError(f"Unexpected jacobians shape: {jacobians.shape}")
           
            # Extract only the columns for controllable DOFs 
            # This gives us the end-effector Jacobian w.r.t. only the arm joints
            J = J_full[:, self.joint_pos_addrs]

            # Check for NaN or inf values
            if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                raise RuntimeError("Jacobian contains NaN or infinite values")
            
            # Assign to internal storage
            # Linear velocity Jacobian (first 3 rows)
            self._J6N[:3, :] = J[:3, :]
            # Angular velocity Jacobian (last 3 rows)  
            self._J6N[3:, :] = J[3:, :]
            
            # Additional validation - check if Jacobian makes sense (non-zero values)
            if np.allclose(J, 0):
                raise RuntimeError("Jacobian is all zeros - check robot configuration and joint addresses")
            
        elif object_type == "geom":
            raise NotImplementedError("Calculating the Jacobian for 'geom' is not yet implemented.")
        elif object_type == "site":
            raise NotImplementedError("Calculating the Jacobian for 'site' is not yet implemented.")
        else:
            raise ValueError(f"Invalid object type specified: {object_type}")
        
        return np.copy(self._J6N)
        




    def M(self, q=None):
        """
        Returns the inertia matrix for the controlled arm_joints.
        Args:
            q (np.ndarray, optional): Joint positions
            indices (optional): Specific articulation indices
            
        Returns:
            np.ndarray: Inertia matrix
        """
        # get_mass_matrices returns (num_envs, dof_count, dof_count)
        M = self.articulation_view.get_mass_matrices()
        # If you have only one robot in ArticulationView
        M_full = M[0]
        # extract only the controlled DOF
        M_arm = M_full[np.ix_(self.joint_pos_addrs, self.joint_pos_addrs)]
        return np.copy(M_arm)


    def R(self, name, q=None, object_type="body"):
        """
        Returns the rotation matrix of the specified object in Isaac Sim.
        
        Parameters
        ----------
        name : str
            The name of the object (body, geom, or site) in the USD stage.
        q : np.ndarray, optional
            Joint positions (not used here unless you want to simulate a different state).
        object_type : str
            One of "body", "geom", or "site".
        """

        if object_type == "body":
            prim_path = self._get_prim_path(name)
        elif object_type in ["site", "geom"]:
            # Assume full path is given or fixed base path
            prim_path = f"{self.prim_path}/{name}"
        else:
            raise ValueError(f"Unsupported object type: {object_type}")
        
        
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Prim '{prim_path}' not found")
                
        # Get 4x4 world transform matrix
        matrix = omni.usd.get_world_transform_matrix(prim)

        # Convert to 3x3 rotation matrix using numpy
        R = np.array([
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
            [matrix[2][0], matrix[2][1], matrix[2][2]]
        ])

        return R
            
        

    def quaternion(self, name, q=None):
        """Returns the quaternion of the specified body
        Parameters
        ----------

        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        if name == "EE": name = self.ee_link_name

        prim_path = self._get_prim_path(name)
        prim = self.stage.GetPrimAtPath(prim_path)

        # Get 4x4 world transform matrix
        matrix = omni.usd.get_world_transform_matrix(prim)
        quat = matrix.ExtractRotationQuat()  

        # Convert to [w, x, y, z] NumPy array 
        quat_np = np.array([quat.GetReal(), *quat.GetImaginary()])
      
        return quat_np

    def C(self, q=None, dq=None):
        """NOTE: The Coriolis and centrifugal effects (and gravity) are
        already accounted for by Mujoco in the qfrc_bias variable. There's
        no easy way to separate these, so all are returned by the g function.
        To prevent accounting for these effects twice, this function will
        return an error instead of qfrc_bias again.
        """
        raise NotImplementedError(
            "Coriolis and centrifugal effects already accounted "
            + "for in the term return by the gravity function."
        )

    def T(self, name, q=None, x=None):
        """Get the transform matrix of the specified body.

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        raise NotImplementedError


    def Tx(self, name, q=None, x=None, object_type="body"):
        """Simplified version that only gets current position without state changes."""
        if name == "EE": name = self.ee_link_name
        # Get prim path
        if object_type in ["body", "link"]:
            prim_path = self._get_prim_path(name)
        elif object_type == "joint":
            # For joints, you might want the parent link position
            prim_path = self._get_prim_path(name)
        else:
            raise ValueError(f"Unsupported object_type: {object_type}")
    
        # Get world position
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim at path: {prim_path}")
        
        matrix = omni.usd.utils.get_world_transform_matrix(prim)
        position = matrix.ExtractTranslation()

        return np.array([position[0], position[1], position[2]], dtype=np.float64)


    def T_inv(self, name, q=None, x=None):
        """Get the inverse transform matrix of the specified body.

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        raise NotImplementedError
    

    # HELPER FUNCTIONS
    # get the prim path for the name of the link, joint, or site
    def _get_prim_path(self, name):
        for prim in self.stage.Traverse():
            #TODO could be more general to inlcude TCP etc
            if str(prim.GetPath()).endswith(name):
                return prim.GetPath()
        return None
            

    def get_joint_positions(self):
        """Get current joint positions"""
        if hasattr(self, 'articulation'):
            all_positions = self.articulation.get_joint_positions()
            return all_positions[self.joint_vel_addrs]
        return None

    def get_joint_velocities(self):
        """Get current joint velocities"""
        if hasattr(self, 'articulation'):
            all_velocities = self.articulation.get_joint_velocities()
            return all_velocities[self.joint_vel_addrs]
        return None
 

    def set_joint_positions(self, q):
        """Set joint positions"""
        if hasattr(self, 'articulation'):
            full_positions = self.articulation.get_joint_positions()
            for i, idx in enumerate(range(len(q))):
                full_positions[idx] = q[i]
            self.articulation.set_joint_positions(full_positions)
    

    def set_joint_velocities(self, dq):
        """Set joint velocities"""
        if hasattr(self, 'articulation'):
            full_velocities = self.articulation.get_joint_velocities()
            for i, idx in enumerate(range(len(dq))):
                full_velocities[idx] = dq[i]
            self.articulation.set_joint_velocities(full_velocities)

 