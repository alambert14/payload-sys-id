import numpy as np


# DH parameters
from pydrake.math import RollPitchYaw

DH_params = [
    {'alpha': -np.pi / 2, 'd': 0.340, 'a': 0},
    {'alpha': np.pi / 2, 'd': 0.0, 'a': 0},
    {'alpha': np.pi / 2, 'd': 0.400, 'a': 0},
    {'alpha': -np.pi / 2, 'd': 0.0, 'a': 0},
    {'alpha': -np.pi / 2, 'd': 0.400, 'a': 0},
    {'alpha': np.pi / 2, 'd': 0.0, 'a': 0},
    {'alpha': 0, 'd': 0.126, 'a': 0},
]


def ik(X_W, init_q):
    position = X_W.translation()
    rotation = X_W.rotation().ToQuaternion().wxyz()


# def ik(X_W, init_q):
#     p_eef = X_W.translation()
#     R_eef = X_W.rotation()
#     rpy_eef = RollPitchYaw(R_eef)
#
#     p_shoulder = np.array([0, 0, DH_params[0]['d']])
#
#     e_z = np.array([0, 0, 1])
#     p_wrist = p_eef - (rpy_eef[1] * rpy_eef[2] * e_z * DH_params[6]['d'])
#
#     # is P_c (vector from shoulder to circle radius) arbitary?
#
#     # Maybe this is just like a nice idea but not actually necessary
#
#
#     # DoDifferentialInverseKinematics


