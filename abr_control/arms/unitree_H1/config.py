import numpy as np
import sympy as sp

from ..base_config import BaseConfig


class Config(BaseConfig):
    """Robot config file for the right H1 arm for IsaacSim

    Attributes
    ----------
    START_ANGLES : numpy.array
        The joint angles for a safe home or rest position
    _M_LINKS : sympy.diag
        inertia matrix of the links
    _M_JOINTS : sympy.diag
        inertia matrix of the joints
    L : numpy.array
        segment lengths of arm [meters]
    KZ : sympy.Matrix
        z isolation vector in orientational part of Jacobian

    Transform Naming Convention: Tpoint1point2
    ex: Tj1l1 transforms from joint 1 to link 1

    Transforms are broken up into two matrices for simplification
    ex: Tj0l1a and Tj0l1b where the former transform accounts for
    joint rotations and the latter accounts for static rotations
    and translations
    """

    def __init__(self, **kwargs):

        super().__init__(N_JOINTS=4, N_LINKS=5, ROBOT_NAME="unitree_H1", **kwargs)

        self._T = {}  # dictionary for storing calculated transforms
        
        self.JOINT_NAMES = [f"H1_right_arm_joint{ii}" for ii in range(self.N_JOINTS)]

        # for the null space controller, keep arm near these angles
        #self.START_ANGLES = np.array(
        #    [np.pi / 4.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0], dtype="float32"
        #)
        self.START_ANGLES = np.array(
            [0.0,  0.0,  0.0,  0.0], dtype="float32"
        )

	# TODO adapt this
        # create the inertia matrices for each link of the three joint
        # TODO: identify the actual values for these links
        self._M_LINKS.append(sp.zeros(6, 6))  # link0
        self._M_LINKS.append(sp.diag(0.001, 0.0, 0.0, 0.001, 0.0, 0.001))   # link1 - right_shoulder_pitch_link
        self._M_LINKS.append(sp.diag(0.002, 0.0, 0.0, 0.002, 0.0, 0.001))   # link2 - right_shoulder_roll_link
        self._M_LINKS.append(sp.diag(0.004, 0.0, 0.0, 0.004, 0.0, 0.0))     # link3 - right_shoulder_yaw_link
        self._M_LINKS.append(sp.diag(0.0, 0.0, 0.0, 0.006, 0.0, 0.006))        # link4 - right_elbow_link

        # the joints don't weigh anything
        self._M_JOINTS = [sp.zeros(6, 6) for ii in range(self.N_JOINTS)]

        # segment lengths associated with each joint
        # [x, y, z],  Ignoring lengths < 1e-04

        
        self.L = [
            [0.0, 0.0, 0.0],                        # base offset (optional)
            [0.0055, -0.15535, 0.42999],            # right_shoulder_pitch_joint
            [0.005045, -0.053657, -0.015715],       # right_shoulder_pitch_link
            [-0.0055, -0.0565, -0.0165],            # right_shoulder_roll_joint
            [0.000679, -0.00115, -0.094076],        # right_shoulder_roll_link
            [0.0, 0.0, -0.1343],                    # right_shoulder_yaw_joint
            [0.01365, -0.002767, -0.16266],         # right_shoulder_yaw_link
            [0.0185, 0.0, -0.198],                  # right_elbow_joint
            [0.164862, -0.000118, -0.015734],       # right_elbow_link
            [0.0, 0.0, 0.09],                       # offset to end of hand/fingers (assumed)
        ]
        self.L = np.array(self.L)



        '''
        self.L = [
            [0.0, 0.0, 0.0],                        # base offset (optional, can be torso_link)
            [0.0055, -0.15535, 0.42999],            # right_shoulder_pitch_joint (from torso_link)
            [0.005045, -0.053657, -0.015715],       # right_shoulder_pitch_link (inertial origin)
            [-0.0055, -0.0565, -0.0165],            # right_shoulder_roll_joint (from shoulder_pitch_link)
            [0.000679, -0.00115, -0.094076],        # right_shoulder_roll_link (inertial origin)
            [0.0, 0.0, -0.1343],                    # right_shoulder_yaw_joint (from shoulder_roll_link)
            [0.01365, -0.002767, -0.16266],         # right_shoulder_yaw_link (inertial origin)
            [0.0185, 0.0, -0.198],                  # right_elbow_joint (from shoulder_yaw_link)
            [0.164862, -0.000118, -0.015734],       # right_elbow_link (inertial origin)
        ]
        '''



        # Transform matrix : origin -> link 0
        # no change of axes, account for offsets
        self.Torgl0 = sp.Matrix(
            [
                [1, 0, 0, self.L[0, 0]],
                [0, 1, 0, self.L[0, 1]],
                [0, 0, 1, self.L[0, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix : link 0 -> joint 0
        # no change of axes, account for offsets
        self.Tl0j0 = sp.Matrix(
            [
                [1, 0, 0, self.L[1, 0]],
                [0, 1, 0, self.L[1, 1]],
                [0, 0, 1, self.L[1, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix : joint 0 -> link 1
        # account for rotation of q
        self.Tj0l1a = sp.Matrix(
            [
                [sp.cos(self.q[0]), -sp.sin(self.q[0]), 0, 0],
                [sp.sin(self.q[0]), sp.cos(self.q[0]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # no change of axes, account for offsets
        self.Tj0l1b = sp.Matrix(
            [
                [1, 0, 0, self.L[2, 0]],
                [0, 1, 0, self.L[2, 1]],
                [0, 0, 1, self.L[2, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj0l1 = self.Tj0l1a * self.Tj0l1b

        # Transform matrix : link 1 -> joint 1
        # no change of axes, account for offsets
        self.Tl1j1 = sp.Matrix(
            [
                [1, 0, 0, self.L[3, 0]],
                [0, 1, 0, self.L[3, 1]],
                [0, 0, 1, self.L[3, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix : joint 1 -> link 2
        # account for rotation of q
        self.Tj1l2a = sp.Matrix(
            [
                [sp.cos(self.q[1]), -sp.sin(self.q[1]), 0, 0],
                [sp.sin(self.q[1]), sp.cos(self.q[1]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # no change of axes, account for offsets
        self.Tj1l2b = sp.Matrix(
            [
                [1, 0, 0, self.L[4, 0]],
                [0, 1, 0, self.L[4, 1]],
                [0, 0, 1, self.L[4, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj1l2 = self.Tj1l2a * self.Tj1l2b

        # Transform matrix : link 2 -> joint 2
        # no change of axes, account for offsets
        self.Tl2j2 = sp.Matrix(
            [
                [1, 0, 0, self.L[5, 0]],
                [0, 1, 0, self.L[5, 1]],
                [0, 0, 1, self.L[5, 2]],
                [0, 0, 0, 1],
            ]
        )

        # Transform matrix : joint 2 -> link 3
        # account for rotation of q
        self.Tj2l3a = sp.Matrix(
            [
                [sp.cos(self.q[2]), -sp.sin(self.q[2]), 0, 0],
                [sp.sin(self.q[2]), sp.cos(self.q[2]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # no change of axes, account for offsets
        self.Tj2l3b = sp.Matrix(
            [
                [1, 0, 0, self.L[6, 0]],
                [0, 1, 0, self.L[6, 1]],
                [0, 0, 1, self.L[6, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj2l3 = self.Tj2l3a * self.Tj2l3b

        # Transform matrix : link 3 -> end-effector
        # no change of axes, account for offsets
        self.Tl3j3 = sp.Matrix(
            [
                [1, 0, 0, self.L[7, 0]],
                [0, 1, 0, self.L[7, 1]],
                [0, 0, 1, self.L[7, 2]],
                [0, 0, 0, 1],
            ]
        )
        
         # Transform matrix : joint 3 -> link 4
        # account for rotation of q
        self.Tj3l4a = sp.Matrix(
            [
                [sp.cos(self.q[3]), -sp.sin(self.q[3]), 0, 0],
                [sp.sin(self.q[3]), sp.cos(self.q[3]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # no change of axes, account for offsets
        self.Tj3l4b = sp.Matrix(
            [
                [1, 0, 0, self.L[8, 0]],
                [0, 1, 0, self.L[8, 1]],
                [0, 0, 1, self.L[8, 2]],
                [0, 0, 0, 1],
            ]
        )
        self.Tj3l4 = self.Tj3l4a * self.Tj3l4b

        # Transform matrix : link 4 -> end-effector
        # no change of axes, account for offsets
        self.Tl4ee = sp.Matrix(
            [
                [1, 0, 0, self.L[9, 0]],
                [0, 1, 0, self.L[9, 1]],
                [0, 0, 1, self.L[9, 2]],
                [0, 0, 0, 1],
            ]
        )


        # orientation part of the Jacobian (compensating for angular velocity)
        self.J_orientation = [
            self._calc_T("joint0")[:3, :3] * self._KZ,  # joint 0 orientation
            self._calc_T("joint1")[:3, :3] * self._KZ,  # joint 1 orientation
            self._calc_T("joint2")[:3, :3] * self._KZ,	# joint 2 orientation
            self._calc_T("joint3")[:3, :3] * self._KZ,
        ]  # joint 3 orientation

    def _calc_T(self, name):
        """Uses Sympy to generate the transform for a joint or link

        name : string
            name of the joint, link, or end-effector
        """

        if self._T.get(name, None) is None:
            if name == "link0":
                self._T[name] = self.Torgl0
            elif name == "joint0":
                self._T[name] = self._calc_T("link0") * self.Tl0j0
            elif name == "link1":
                self._T[name] = self._calc_T("joint0") * self.Tj0l1
            elif name == "joint1":
                self._T[name] = self._calc_T("link1") * self.Tl1j1
            elif name == "link2":
                self._T[name] = self._calc_T("joint1") * self.Tj1l2
            elif name == "joint2":
                self._T[name] = self._calc_T("link2") * self.Tl2j2
            elif name == "link3":
                self._T[name] = self._calc_T("joint2") * self.Tj2l3
            elif name == "joint3":
                self._T[name] = self._calc_T("link3") * self.Tl3j3
            elif name == "link4":
                self._T[name] = self._calc_T("joint3") * self.Tj3l4
            elif name == "EE":
                self._T[name] = self._calc_T("link4") * self.Tl4ee

            else:
                raise Exception(f"Invalid transformation name: {name}")

        return self._T[name]
