import os
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
    



    def __init__(self, xml_file, folder=None, use_sim_state=True, force_download=False):
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
        if folder is None:
            arm_dir = xml_file.split("_")[0]
            current_dir = os.path.dirname(__file__)
            self.xml_file = os.path.join(current_dir, arm_dir, f"{xml_file}.xml")
            self.xml_dir = f"{current_dir}/{arm_dir}"
        else:
            self.xml_dir = f"{folder}"
            self.xml_file = os.path.join(self.xml_dir, xml_file)

        self.N_GRIPPER_JOINTS = 0

        # get access to some of our custom arm parameters from the xml definition
        tree = ElementTree.parse(self.xml_file)
        root = tree.getroot()
        for custom in root.findall("custom/numeric"):
            name = custom.get("name")
            if name == "START_ANGLES":
                START_ANGLES = custom.get("data").split(" ")
                self.START_ANGLES = np.array([float(angle) for angle in START_ANGLES])
            elif name == "N_GRIPPER_JOINTS":
                self.N_GRIPPER_JOINTS = int(custom.get("data"))

        # check for google_id specifying download location of robot mesh files
        self.google_id = None
        for custom in root.findall("custom/text"):
            name = custom.get("name")
            if name == "google_id":
                self.google_id = custom.get("data")

        actuators = root.find(f'actuator')
        self.joint_names = [actuator.get("joint") for actuator in actuators]
     
        # check if the user has downloaded the required mesh files
        # if not prompt them to do so
        if self.google_id is not None:
            # get list of expected files to check if all have been downloaded
            files = []
            for asset in root.findall("asset/mesh"):
                files.append(asset.get("file"))

            for asset in root.findall("asset/texture"):
                # assuming that texture are placed in the meshes folder
                files.append(asset.get("file").split("/")[1])

            # check if our mesh folder exists, then check we have all the files
            download_meshes.check_and_download(
                name=self.xml_dir + "/meshes",
                google_id=self.google_id,
                force_download=force_download,
                files=files,
            )

        #TODO fix above to get away from xml
        self.robot = xml_file
        self.use_sim_state = use_sim_state

    def _connect(self, world, stage, articulation, articulation_view, joint_pos_addrs, joint_vel_addrs, prim_path, ee_link_name):

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
        self.ee_link_name = ee_link_name

        
        self.N_JOINTS = len(self.joint_vel_addrs)
        print (f"Number of controllable joints: {self.N_JOINTS}")
        # number of joints in the IsaacSim simulation
        N_ALL_JOINTS = self.articulation_view.num_dof
        print (f"Number of ALL joints in simulation: {N_ALL_JOINTS}")

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
        Returns the joint-space forces due to gravity, Coriolis, and centrifugal effects
        in Isaac Sim (equivalent to MuJoCo's qfrc_bias).

        Parameters
        ----------
        q: np.ndarray, optional (Default: None)
            Joint positions to compute the bias forces at. If None, uses current sim state.
        """
        # Compute gravity and Coriolis/centrifugal separately
        gravity = self.articulation_view.get_generalized_gravity_forces()
        coriolis = self.articulation_view.get_coriolis_and_centrifugal_forces()
        
        # Total generalized bias forces
        g_full = gravity + coriolis
        
        # Handle batch dimension if present
        if g_full.ndim == 2:
            if g_full.shape[0] == 1:
                g_full = g_full[0]  # Remove batch dimension
        
        if q is not None:
            # If q is provided, ensure g matches the size of q
            if len(g_full) != len(q):
                g_full = g_full[:len(q)]
        
        return -g_full
        
    

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
        """Returns the Jacobian for the specified link,
        computed at the origin of the link's rigid body frame,
        which in Isaac Sim coincides with the link's center of mass.
        
        Parameters
        ----------
        name: string
            The name of the link/body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Isaac Sim simulator
        x: float numpy.array, optional (Default: None)
            Offset from link origin (currently not supported)
        object_type: string, optional (Default: "body")
            The object type - options: body, geom, site
        controlled_dofs: list, optional (Default: None)
            List of DOF indices to include in Jacobian. If None, uses first 6 DOFs
        
        Returns
        -------
        numpy.array
            6xN Jacobian matrix where N is number of controlled DOFs
            First 3 rows: linear velocity Jacobian (Jv)
            Last 3 rows: angular velocity Jacobian (Jω)
        """
        if name == "EE": name = self.ee_link_name
        
        # Check for unsupported features
        if x is not None and not np.allclose(x, 0):
            raise Exception("x offset currently not supported, set to None")
        
        
        if object_type == "body":
            # Get Jacobians from articulation view
            jacobians = self.articulation_view.get_jacobians(clone=True)
                
            # Check if ArticulationView has any environments
            if jacobians.shape[0] == 0:
                raise RuntimeError("ArticulationView contains no environments. Make sure it's properly initialized and contains articulations.")
                
            
            #TODO remove of fix method used for UR5
            #link_index = self._get_link_index(name) 
            # Get link index
            link_index = self.articulation_view.get_link_index(name)
           
            # Extract Jacobian for specific link
            env_idx = 0  # Assuming single environment
                
            # Handle different Jacobian tensor shapes
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
            J = J_full[:, self.joint_pos_addrs]

            # Verify dimensions
            if J.shape[0] != 6:
                raise RuntimeError(f"Expected Jacobian to have 6 rows, got {J.shape[0]}")
            if J.shape[1] != len(self.joint_pos_addrs):
                raise RuntimeError(f"Expected Jacobian to have {len(self.joint_pos_addrs)} columns, got {J.shape[1]}")
                
            # Assign to internal storage
            # Linear velocity Jacobian (first 3 rows)
            self._J6N[:3, :] = J[:3, :]
            # Angular velocity Jacobian (last 3 rows)
            self._J6N[3:, :] = J[3:, :]
                
        elif object_type == "geom":
            raise NotImplementedError("Calculating the Jacobian for 'geom' is not yet implemented.")
        elif object_type == "site":
            raise NotImplementedError("Calculating the Jacobian for 'site' is not yet implemented.")
        else:
            raise ValueError(f"Invalid object type specified: {object_type}")
        
        return np.copy(self._J6N)



    
    def M(self, q=None, indices=None):
        """
        Returns the inertia matrix using Isaac Sim Core API
        
        Args:
            q (np.ndarray, optional): Joint positions
            indices (optional): Specific articulation indices
            
        Returns:
            np.ndarray: Inertia matrix
        """
        M = self._compute_mass_matrix_numerical()
    
        # Restore original state if q was provided
        #if q is not None:
        #    self.articulation.set_joint_positions(current_positions)
        #    self.world.step(render=False)
        
        return np.copy(M)


    #TODO: CHECK AND IF NECEAARY remove this, use M instead
    def _compute_mass_matrix_numerical(self):
        """
        Compute mass matrix numerically using finite differences
        This is a fallback method when direct mass matrix access is not available
        """
        import numpy as np
        
        M = np.zeros((self.N_JOINTS, self.N_JOINTS))
        
        # Small perturbation for finite differences
        epsilon = 1e-6
        
        # Get current state
        current_pos = self.get_joint_positions()
        current_vel = self.get_joint_velocities()

        # Zero velocities for clean computation
        zero_vel = np.zeros_like(current_vel)
        self.set_joint_velocities(zero_vel)
        
        # Compute each column of mass matrix
        for i in range(self.N_JOINTS):
            # Create unit acceleration in joint i
            unit_accel = np.zeros(self.N_JOINTS)
            unit_accel[i] = 1.0
            
            # Compute required torques for this acceleration
            # Using inverse dynamics: tau = M*qdd + C + G
            tau = self._compute_inverse_dynamics(current_pos, zero_vel, unit_accel)
            
            # The torques give us the i-th column of the mass matrix
            M[:, i] = tau
        
        return M
      


    def _compute_inverse_dynamics(self, q, qd, qdd):
        """
        Compute inverse dynamics: tau = M*qdd + C + G
        This is an approximation using IsaacSim's physics engine
        """
        '''
        # Method 1: Use PhysX articulation dynamics (if available)
        try:
            # Set desired accelerations and compute required forces
            full_accelerations = np.zeros(self.robot.num_dof)
            for i, idx in enumerate(self.joint_vel_addr):
                full_accelerations[idx] = qdd[i]
            
            # Use articulation's compute_efforts method if available
            if hasattr(self.robot, 'compute_efforts'):
                efforts = self.robot.compute_efforts(
                    positions=self.robot.get_joint_positions(),
                    velocities=self.robot.get_joint_velocities(),
                    accelerations=full_accelerations
                )
                return efforts[self.joint_vel_addr]
        except:
            pass
        '''
        # Method 2: Numerical approximation
        # Apply accelerations and measure required torques
        dt = self.world.get_physics_dt()
        
        # Store current state
        original_pos = self.get_joint_positions()
        original_vel = self.get_joint_velocities()
        
        # Set desired state
        self.set_joint_positions(q)
        self.set_joint_velocities(qd)
        
        # Compute target velocities after acceleration
        target_vel = qd + qdd * dt
        
        # Use PD control to estimate required torques
        kp = 1000.0  # High proportional gain
        kd = 100.0   # Damping
        
        pos_error = np.zeros_like(q)  # No position error
        vel_error = target_vel - qd
        
        tau = kp * pos_error + kd * vel_error
        
        # Restore original state
        self.set_joint_positions(original_pos)
        self.set_joint_velocities(original_vel)
        
        return tau



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
        '''
        
        print(f"R: {name}, q: {q}, object_type: {object_type}")
        if not self.use_sim_state and q is not None:
            old_q = self.articulation_view.get_joint_positions()
            self.articulation_view.set_joint_positions(q)
            #self._world.step(render=False)
        '''

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

        from pxr import Gf
        # Extract quaternion (as Gf.Quatf or Gf.Quatd)
        quat = matrix.ExtractRotationQuat()  # returns Gf.Quatd or Gf.Quatf

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
        #print(f"Tx: name={name}, q={q}, x={x}, object_type={object_type}")  
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
    '''
    def set_joint_positions(self, positions):
        """Set joint positions"""
        if hasattr(self, 'articulation'):
            full_positions = self.articulation.get_joint_positions()
            for i, idx in enumerate(range(len(positions))):
                full_positions[idx] = positions[i]
            self.articulation.set_joint_positions(full_positions)

    '''
    def set_joint_positions(self, positions):
        """Set joint positions"""
        if hasattr(self, 'articulation'):
            full_positions = self.articulation.get_joint_positions()
            for i, idx in enumerate(self.joint_vel_addrs):
                full_positions[idx] = positions[i]
            self.articulation.set_joint_positions(full_positions)
    

    def set_joint_velocities(self, velocities):
        """Set joint velocities"""
        if hasattr(self, 'articulation'):
            full_velocities = self.articulation.get_joint_velocities()
            for i, idx in enumerate(self.joint_vel_addrs):
                full_velocities[idx] = velocities[i]
            self.articulation.set_joint_velocities(full_velocities)

    #TODO remove of fix method used for UR5
    def _get_link_index(self, name):
        """Get the index of a link by its name"""

        import omni.isaac.core.utils.prims as prims_utils
        #import omni.isaac.core.prims as prims_utils

        prim_path = self._get_prim_path(name)
        prim = prims_utils.get_prim_at_path(prim_path)
        parent_prim = prims_utils.get_prim_parent(prim)
        parent_name = parent_prim.GetName()
        link_index = self.articulation_view.get_link_index(parent_name)
        print(f"Link name: {name}, Prim path: {prim_path}, Parent name: {parent_name}, Link index: {link_index}")
        
        return link_index + 1