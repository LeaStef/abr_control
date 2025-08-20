import numpy as np
import omni


class IsaacsimConfig:
    """A wrapper on the IsaacSim simulator to generate all the kinematics and
    dynamics calculations necessary for controllers.
    """

    def __init__(self, robot_type):
        """Loads the Isaacsim model from the specified xml file

        Parameters
        ----------
        robot_type: string
            the name of the arm model to load. 
        """
        self.robot_type = robot_type
        self.env_idx = 0  # Assuming only one robot
        self.ee_link_name = "EE"  # name used for virtual end-effector, overwritten if one exists already
        
        if self.robot_type == "ur5":
            self.robot_path = "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
            self.has_EE = False  # UR5 has no end-effector
            self.EE_parent_link = "wrist_3_link"  
            self.ee_offset=[0., 0., 0.] 
            START_ANGLES = "0 -.67 -.67 0 0 0"
            self.target_min = np.array([-0.4, -0.4, 0.3]) # np.array([-0.6, -0.5, 0.5])
            self.target_max = np.array([0.4, 0.4, 0.6]) 
            self.controlled_dof = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            print(f"Virtual end effector with name '{self.ee_link_name}' is attached, as robot has none.")

        elif self.robot_type == "jaco2":
            self.robot_path = "/Isaac/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd"
            self.has_EE = True  # jaco2 has an end-effector
            self.EE_parent_link = "j2n6s300_link_6"  
            self.ee_link_name = "j2n6s300_end_effector"  
            START_ANGLES = "2.0 3.14 1.57 4.71 0.0 3.04"
            self.target_min = np.array([-0.4, -0.4, 0.3]) # np.array([-0.5, -0.5, 0.5])
            self.target_max = np.array([0.4, 0.4, 0.6]) 
            self.controlled_dof = ['j2n6s300_joint_1', 'j2n6s300_joint_2', 'j2n6s300_joint_3', 'j2n6s300_joint_4', 'j2n6s300_joint_5', 'j2n6s300_joint_6']
            print(f"End effector with name '{self.ee_link_name}' specified in UDS, using it ...")
                
        elif self.robot_type == "h1":   
            self.robot_path = "/Isaac/Robots/Unitree/H1/h1.usd"
            self.has_EE = False  # H1 has no end-effector
            self.EE_parent_link = "right_elbow_link"
            self.ee_offset=[0.26455, 0.00118, -0.0209] # for H1 
            START_ANGLES = "0. 0. 0. 0."
            self.target_min = np.array([0.12, -0.4, 1.4])
            self.target_max = np.array([0.4, 0.05, 2.2]) 
            self.controlled_dof = ['right_shoulder_pitch_joint','right_shoulder_roll_joint', 'right_shoulder_yaw_joint','right_elbow_joint']
            self.lock_prim_standing = '/World/robot/pelvis'
            print(f"Virtual end effector with name '{self.ee_link_name}' is attached, as robot has none.")

        elif self.robot_type == "h1_hands":  
            self.robot_path = "/Isaac/Robots/Unitree/H1/h1_with_hand.usd"
            self.has_EE = True  # H1 with hands has an end-effector
            self.EE_parent_link = "right_elbow_link"
            self.ee_link_name = "right_hand_link"  
            START_ANGLES = "0. 0. 0. 0. 0."
            self.target_min = np.array([0.12, -0.4, 1.4])
            self.target_max = np.array([0.49, 0.05, 2.2]) 
            self.controlled_dof = ['right_shoulder_pitch_joint','right_shoulder_roll_joint', 'right_shoulder_yaw_joint','right_elbow_joint','right_hand_joint']
            self.lock_prim_standing = '/World/robot/pelvis'
            print(f"End effector with name '{self.ee_link_name}' specified in UDS, using it ...")

        self.START_ANGLES = np.array(START_ANGLES.split(), dtype=float)

        

    def _connect(self, world, stage, articulation_view, dof_indices, joint_indices, prim_path):
        """Called by the interface once the IsaacSim simulation is created,
    this connects the config to the simulator so it can access the
    kinematics and dynamics information calculated by IsaacSim.
    
    Parameters
    ----------
    world : omni.isaac.core.World
        The Isaac Sim World object that manages the simulation environment
    stage : pxr.Usd.Stage
        The USD stage containing the scene hierarchy and prims
    articulation_view : omni.isaac.core.articulations.ArticulationView
        The ArticulationView object that provides access to the robot's
        kinematic and dynamic properties in Isaac Sim
    dof_indices : np.array of ints
        The indices of the controlled robot joints in the articulation's
        DOF array. These correspond to the joints that will be controlled
        and for which kinematics/dynamics will be computed.
    joint_indices : np.array of ints
        The indices of the controlled robot joints in the articulation's
        joint array. Note that joint_indices and dof_indices may differ
        due to fixed joints or different indexing schemes.
    prim_path : str
        The USD prim path to the robot articulation in the scene hierarchy
        (e.g., "/World/Robot")
    """
        # get access to IsaacSim
        self.world = world
        self.stage = stage
        self.articulation_view = articulation_view
        self.dof_indices = np.copy(dof_indices)
        self.joint_indices = np.copy(joint_indices)
        self.prim_path = prim_path
        self.N_JOINTS = len(self.dof_indices)
        self.N_ALL_DOF = self.articulation_view.num_dof
        self.N_ALL_JOINTS = self.articulation_view.num_joints
        self._is_fixed_base = self._is_fixed_base() 
        # offset for non-fixed base (mobile robots), necessary for indexing jacobian matrices etc correctly
        self.base_offset = 6 if not self._is_fixed_base else 0

        self.jac_indices = np.hstack(
            # 6 because position and rotation Jacobians are 3 x N_ALL_DOF
            [self.dof_indices + (ii * self.N_ALL_DOF) for ii in range(3)]
        )

        # for the inertia matrix
        self.M_indices = [
            ii * self.N_ALL_DOF + jj
            for jj in self.dof_indices
            for ii in self.dof_indices
        ]

        # a place to store data returned from IsaacSim
        self._g = np.zeros(self.N_JOINTS)
        #self._J3NP = np.zeros((3, self.N_ALL_DOF)) # TODO
        #self._J3NR = np.zeros((3, self.N_ALL_DOF)) # TODO
        self._J6N = np.zeros((6, self.N_JOINTS)) 
        self._MNN = np.zeros((self.N_ALL_DOF, self.N_ALL_DOF))
        #self._R9 = np.zeros(9)
        self._R = np.zeros((3, 3))
        #self._x = np.ones(4)


    def g(self, q=None):
        """Returns gravitational forces.

        Parameters
        ----------
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. 
        """
        self._g = self.articulation_view.get_generalized_gravity_forces()[self.env_idx]
        if self._is_fixed_base:
            return -self._g[self.dof_indices]
        else:
            return np.zeros(len(self.dof_indices))
            

    def dJ(self, name, q=None, dq=None, x=None):
        """Returns the derivative of the Jacobian wrt to time

        Parameters
        ----------
        name: string
            The name of the IsaacSim prim to retrieve the derivative of the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. 
        dq: float numpy.array, optional (Default: None)
            The joint velocities of the robot. 
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        raise NotImplementedError


    def J(self, name, q=None, x=None):
        """Returns the Jacobian for the controlled DOF.
        In case of mobile robots the floating base is accounted for by an offset.

        Parameters
        ----------
        name: string
            The name of the IsaacSim prim to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. 
        x: float numpy.array, optional (Default: None)
        """
        jacobians = self.articulation_view.get_jacobians(clone=True)
        if jacobians.shape[self.env_idx] == 0:
            raise RuntimeError("ArticulationView contains no environments.")

        offset = 0 if not self._is_fixed_base else -1
        body_index = self.articulation_view.get_body_index(self.EE_parent_link) + offset
      
        
        J_full = jacobians[self.env_idx, body_index, :, :]
        if len(jacobians.shape) == 4:
            # apply offset for non-fixed base (mobile robots)
            indices = self.dof_indices + self.base_offset
            # Extract only the columns for controllable DOFs 
            J = J_full[:, indices]
        else:
            raise RuntimeError(f"Unexpected jacobians shape: {jacobians.shape}")
            
        # Linear velocity Jacobian (first 3 rows)
        self._J6N[:3, :] = J[:3, :]
        # Angular velocity Jacobian (last 3 rows)  
        self._J6N[3:, :] = J[3:, :]
            
        if np.allclose(J, 0):
            raise RuntimeError("Jacobian is all zeros - check robot configuration and joint addresses")
        return np.copy(self._J6N)



    def M(self, q=None):
        """Returns the inertia matrix in task space for the controlled DOF.
        In case of mobile robots the floating base is accounted for by an offset.

        Parameters
        ----------
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. 
        """
        # get_mass_matrices returns (num_envs, joint_count, joint_count)
        self._MNN= self.articulation_view.get_mass_matrices()[self.env_idx]
        # apply offset for non-fixed base (mobile robots)
        indices = self.dof_indices + self.base_offset
        M = self._MNN[np.ix_(indices, indices)]
        return M

        
    def R(self, name, q=None):
        """
        Returns the rotation matrix of the specified object in IsaacSim.
        
        Parameters
        ----------
        name : str
            The name of the object (body, geom, or site) in the USD stage.
        q : np.ndarray, optional
            Joint positions (not used here unless you want to simulate a different state).
        """
        prim = self._get_prim(name)
        if not prim.IsValid():
            raise RuntimeError(f"Prim '{prim_path}' not found")
                
        # Get 4x4 world transform matrix
        matrix = omni.usd.get_world_transform_matrix(prim)

        # Convert to 3x3 rotation matrix using numpy
        self._R = np.array([
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
            [matrix[2][0], matrix[2][1], matrix[2][2]]
        ])

        return self._R  
        

    def quaternion(self, name, q=None):
        """Returns the quaternion of the specified prim.
        Parameters
        ----------

        name: string
            The name of the IsaacSim prim to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. 
        """
        if name == "EE": name = self.ee_link_name
        prim = self._get_prim(name)

        # Get 4x4 world transform matrix
        matrix = omni.usd.get_world_transform_matrix(prim)
        quat = matrix.ExtractRotationQuat()  
        quat_np = np.array([quat.GetReal(), *quat.GetImaginary()])
      
        return quat_np


    def C(self):
        """NOTE: The Coriolis and centrifugal effects are neglected atm, as this method is not called anywhere
        """
        coriolis = self.articulation_view.get_coriolis_and_centrifugal_forces(joint_indices=self.dof_indices)[self.env_idx]
        return coriolis


    def T(self, name, q=None, x=None):
        """Returns the transform matrix of the specified prim.

        Parameters
        ----------
        name: string
            The name of the prim to retrieve the transform matrix from.
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot.
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        raise NotImplementedError


    def Tx(self, name, q=None, x=None):
        """Simplified version of T. 
        Returns the position without state changes of the specified prim.

        Parameters
        ----------
        name: string
            The name of the prim to retrieve the position from.
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. 
        x: float numpy.array, optional (Default: None)
        """
        if name == "EE": name = self.ee_link_name
        prim = self._get_prim(name)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim at path: {prim_path}")
        
        matrix = omni.usd.utils.get_world_transform_matrix(prim)
        position = matrix.ExtractTranslation()
        return np.array([position[0], position[1], position[2]], dtype=np.float64)


    def T_inv(self, name, q=None, x=None):
        """Returns the inverse of the transform matrix of the specified prim.

        Parameters
        ----------
        name: string
            The name of the prim to retrieve the inverse from.
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot.
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        raise NotImplementedError
    

    def _get_prim_path(self, name):
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).endswith(name):
                return prim.GetPath()
        return None
    
    def _get_prim(self, name):
        prim_path = self._get_prim_path(name)
        prim = self.stage.GetPrimAtPath(prim_path)
        return prim

            
    def _get_joint_positions(self):
        full_positions = self.articulation_view.get_joint_positions()[self.env_idx]
        return full_positions[self.dof_indices]
    
    def _get_joint_velocities(self):
        full_velocities = self.articulation_view.get_joint_velocities()[self.env_idx]
        return full_velocities[self.dof_indices]

    def _set_joint_positions(self, q):
        full_positions = self.articulation_view.get_joint_positions()[self.env_idx]
        full_positions[self.dof_indices] = q
        self.articulation_view.set_joint_positions(full_positions)
    
    def _set_joint_velocities(self, dq):
        full_velocities = self.articulation_view.get_joint_velocities()[self.env_idx]
        full_velocities[self.dof_indices] = dq
        self.articulation_view.set_joint_velocities(full_velocities)

    def _is_fixed_base(self):
        num_dof = self.articulation_view.num_dof
        jacobian_shape = self.articulation_view.get_jacobian_shape()[2]
        return num_dof == jacobian_shape

