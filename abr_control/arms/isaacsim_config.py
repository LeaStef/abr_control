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

        self.use_sim_state = use_sim_state

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

        
        # number of controllable joints in the robot arm
        #TODO: this is a hack for the jaco2 robot, which has 12 joints
        #self.N_JOINTS = len(self.joint_vel_addrs)
        self.N_JOINTS = self.articulation_view.num_dof
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

    def _load_state(self, q, dq=None, u=None):
        """Change the current joint angles

        Parameters
        ----------
        q: np.array
            The set of joint angles to move the arm to [rad]
        dq: np.array
            The set of joint velocities to move the arm to [rad/sec]
        u: np.array
            The set of joint forces to apply to the arm joints [Nm]
        """

        old_q = np.copy(self.articulation.get_joint_positions())
        old_dq = np.copy(self.articulation.get_joint_velocities())
        old_u = np.copy(self.articulation.get_applied_joint_efforts())

        # update positions to specified state
        self.articulation.set_joint_positions(q)  # set the joint positions in the articulation view

        if dq is not None:
            #self.data.qvel[self.joint_vel_addrs] = np.copy(dq)
            self.articulation_view.set_joint_velocities(dq)
        if u is not None:
            #self.data.ctrl[:] = np.copy(u)
            self.articulation_view.set_joint_efforts(u)

        # move simulation forward to calculate new kinematic information
        self.world.step(render=True) # execute one physics step and one rendering step

        return old_q, old_dq, old_u



    def g(self, q=None):
        """
        Returns the joint-space forces due to gravity, Coriolis, and centrifugal effects
        in Isaac Sim (equivalent to MuJoCo's qfrc_bias).

        Parameters
        ----------
        q: np.ndarray, optional (Default: None)
            Joint positions to compute the bias forces at. If None, uses current sim state.
        """
        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

            # Set new state (velocities to zero to isolate gravity)
            self.articulation_view.set_joint_positions(q)
            self.articulation_view.set_joint_velocities(np.zeros_like(q))

            self.world.step(render=False)

        # Compute gravity and Coriolis/centrifugal separately
        gravity = self.articulation_view.get_generalized_gravity_forces()
        coriolis = self.articulation_view.get_coriolis_and_centrifugal_forces()

        # Total generalized bias forces
        g = gravity + coriolis
        # print("GRAVITY ", gravity)
        # print("CORIOLIS ", coriolis)

        if not q is not None:
            # Restore old state
            self.articulation_view.set_joint_positions(old_q)
            self.articulation_view.set_joint_velocities(old_dq)
            self.world.step(render=False)

        return -g  # match MuJoCo's negative qfrc_bias convention

    

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
        which in Isaac Sim  coincides with the link’s center of mass.

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        object_type: string, the Mujoco object type, optional (Default: body)
            options: body, geom, site
        """
        #TODO outsource this to a common function and check is can be qued for EE and 
        # end_effector at the same time, or checked which is used for the current robot
        if name == "EE": name = "end_effector"

        if x is not None and not np.allclose(x, 0):
            raise Exception("x offset currently not supported, set to None")

        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        if object_type == "body":
            try:
                #TODO maybe outsource if also needed elsewhere
                link_list = self.articulation_view.body_names
                prefix = link_list[0].split('_')[0]  # Get the prefix from the first link name
                full_name = f"{prefix}_{name}"  # Construct the full name
                link_index = link_list.index(full_name)
            except ValueError:
                raise RuntimeError(f"Link '{name}' not found in articulation.")
            jacobians = self.articulation_view.get_jacobians(clone=True)  # shape: (num_links, 6, num_dofs)
            J = jacobians[0, link_index, :, :]


        else:
            if object_type == "geom":
                raise Exception("Calculating the Jacobian is not yet implemented for 'geom'.")
            elif object_type == "site":
                raise Exception("Calculating the Jacobian is not yet implemented for 'site'.")
            else:
                raise Exception("Invalid object type specified: ", object_type)

        # get the position Jacobian (linear velocity Jacobian (Jv))
        self._J6N[:3] = J[:3, :]
        # get the rotation Jacobian (angular velocity Jacobian (Jω))
        self._J6N[3:] = J[3:, :]


        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)

        return np.copy(self._J6N)

    def M(self, q=None):
        """
        Return the joint-space inertia matrix M(q) using Isaac Sim.

        Parameters
        ----------
        q : np.ndarray, optional
            Joint positions. If None, use current sim state.

        Returns
        -------
        np.ndarray
            Dense inertia matrix (DoF x DoF)
        """

        if not self.use_sim_state and q is not None:
            # Save current joint state
            old_q = self.articulation_view.get_joint_positions()
            self.articulation_view.set_joint_positions(q)
            self.world.step(render=False)  # required to update PhysX buffers

        # Get mass matrix
        M = self.articulation_view.get_mass_matrices()
        #print(f"Mass matrix M: {M}")

        if not self.use_sim_state and q is not None:
            # Restore previous state
            self.articulation_view.set_joint_positions(old_q)
            self.world.step(render=False)

        return M

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
        #print(f"R: {name}, q: {q}, object_type: {object_type}")
        if not self.use_sim_state and q is not None:
            old_q = self.articulation_view.get_joint_positions()
            self.articulation_view.set_joint_positions(q)
            #self._world.step(render=False)

        if object_type == "body":
            print(f"Finding body with name: {name}")
            # Look for a link prim ending in the given name
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
        if not self.use_sim_state and q is not None:
            self.articulation_view.set_joint_positions(old_q)
            #self._world.step(render=False)

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
        #TODO outsource this to a common function and check is can be qued for EE and 
        # end_effector at the same time, or checked which is used for the current robot
        if name == "EE": name = "end_effector"
        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        prim_path = self._get_prim_path(name)
        prim = self.stage.GetPrimAtPath(prim_path)

        # Get 4x4 world transform matrix
        matrix = omni.usd.get_world_transform_matrix(prim)

        from pxr import Gf
        # Extract quaternion (as Gf.Quatf or Gf.Quatd)
        quat = matrix.ExtractRotationQuat()  # returns Gf.Quatd or Gf.Quatf

        # Convert to [w, x, y, z] NumPy array 
        quat_np = np.array([quat.GetReal(), *quat.GetImaginary()])

        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)

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
        """ Returns the world-frame Cartesian position of a named link, joint, or site.

        Parameters
        ----------
        name : str
            Name of the link, joint, or site (e.g., "j2n6s300_link_6").
        q : np.ndarray, optional
            Joint positions to temporarily set before computing position.
        object_type : str
            "body" (link), "joint", or custom prim.

        Returns
        -------
        np.ndarray
            World position [x, y, z] of the object.
        """
        if name == "EE": name = "end_effector"
        if x is not None and not np.allclose(x, 0):
            raise Exception("x offset currently not supported: ", x)

        # Optionally set joint state
        if not self.use_sim_state and q is not None:
            old_q = self.articulation_view.get_joint_positions()
            self.articulation_view.set_joint_positions(q)
            self.world.step(render=False)

        prim_path = None
        #print(f"Tx: name={name}, q={q}, object_type={object_type}")

        #TODO similar to others like R, ousource
        if object_type == "body":
            prim_path = self._get_prim_path(name)
            

        elif object_type == "joint":
            # Use joint index to find parent link's prim path
            try:
                joint_index = self.articulation_view.joint_names.index(name)
                link_index = self.articulation_view.joint_parent_indices[joint_index]
                prim_path = self.articulation_view._prim_paths[link_index]
            except ValueError:
                raise RuntimeError(f"Joint name '{name}' not found in articulation.")

        else:
            raise ValueError(f"Unsupported object_type: {object_type}")

        if prim_path is None:
            raise RuntimeError(f"Could not find prim for name '{name}' with type '{object_type}'.")

        # Get world transform matrix
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim at path: {prim_path}")
        
        matrix = omni.usd.utils.get_world_transform_matrix(prim)
        position = matrix.ExtractTranslation()

        # Restore state if needed
        if not self.use_sim_state and q is not None:
            self.articulation_view.set_joint_positions(old_q)
            self.world.step(render=False)
            
        Tx =np.array([position[0], position[1], position[2]])
        return Tx

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
    

    #TODO remove or see if useful
    '''
    def save_current_state(self):
        old_q = np.copy(self.articulation.get_joint_positions())
        old_dq = np.copy(self.articulation.get_joint_velocities())
        old_u = np.copy(self.articulation.get_applied_joint_efforts())

        return old_q, old_dq, old_u
        '''

    #TODO remove or use 
    def restore_state(self, state):
        old_q, old_dq, old_u = state
        self.articulation.set_joint_positions(old_q)
        self.articulation.set_joint_velocities(old_dq)
        self.articulation.set_applied_joint_efforts(old_u)

    def _get_prim_path(self, name):
        for prim in self.stage.Traverse():
            # Check if the prim path ends with the desired name
            if str(prim.GetPath()).endswith(name):
                return prim.GetPath()
        return None
            
