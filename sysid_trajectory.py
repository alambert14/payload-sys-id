from typing import List

import numpy as np

from pydrake.all import (
    DirectCollocation,
    MathematicalProgram,
    MathematicalProgramResult,
    MultibodyPlant,
    PiecewisePolynomial,
    RigidTransform,
    Solve,
)
from pydrake.math import RotationMatrix, RollPitchYaw
from pydrake.multibody.tree import BodyFrame
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.multibody import inverse_kinematics
from manipulation.meshcat_cpp_utils import AddMeshcatTriad
from pydrake.trajectories import PiecewiseQuaternionSlerp
import pydrake.solvers.mathematicalprogram as mp


class SysIDTrajectory:

    def __init__(self, plant, context, initial_state, final_state):
        self.dircol = DirectCollocation(plant,
                                        context,
                                        num_time_samples=21,
                                        minimum_timestep=0.05,
                                        maximum_timestep=0.2)

        self.prog = self.dircol.prog()
        self.dircol.AddEqualTimeIntervalsConstraints()

        # TODO: determine initial and final states
        # Constrain the initial and final states
        self.prog.AddBoundingBoxConstraint(initial_state,
                                           initial_state,
                                           self.dircol.initial_state())
        self.prog.AddBoundingBoxConstraint(final_state,
                                           final_state,
                                           self.dircol.final_state())

        self.dircol.AddRunningCost(self.running_cost)

        # TODO: torque constraints
        u = self.dircol.input()

        self.W = plant.calculate_lumped_parameters()[0]


    def solve(self) -> PiecewisePolynomial:
        result = Solve(self.prog)
        assert (result.is_success())

        u_traj = self.dircol.ReconstructInputTrajectory(result)
        return u_traj

    def running_cost(self, t, q, u):
        """
        :param t: Placeholder decision variables for the time
        :param q: Placeholder decision variables for the state
        :param u: Placeholder decision variables for the commands
        :return: Symbolic expression representing 1 - the condition number of W
        """


## From pangtao/pick-and-place-benchmarking-framework
class SimpleTrajectorySource(LeafSystem):
    def __init__(self, q_traj: PiecewisePolynomial):
        super().__init__()
        self.q_traj = q_traj

        self.x_output_port = self.DeclareVectorOutputPort(
            'x', BasicVector(q_traj.rows()), self.calc_x)

        self.t_start = 0.

    def calc_x(self, context, output):
        t = context.get_time() - self.t_start
        q = self.q_traj.value(t).ravel()
        # v = self.q_traj.derivative(1).value(t).ravel()
        output.SetFromVector(q)  # np.hstack([q, # v]))

    def set_t_start(self, t_start_new: float):
        self.t_start = t_start_new

## Modified from pangtao/pick-and-place-benchmarking-framework SimpleTrajectory
class PickAndPlaceTrajectorySource(LeafSystem):

    def __init__(self, plant: MultibodyPlant, meshcat): # ,
                 # q_init: List[float],
                 # X_WG_start: RigidTransform,
                 # X_WO: RigidTransform,
                 # X_WG_end: RigidTransform, clearance: float = 0.3):
        super().__init__()
        self.set_name('pick_and_place_traj')
        self.plant = plant
        self.meshcat = meshcat
        self.t_start = 0
        self.q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [0, 3], np.vstack([np.zeros(7), np.zeros(7)]).T,
            np.zeros(7), np.zeros(7))

        self.x_output_port = self.DeclareVectorOutputPort(
            'traj_x', BasicVector(self.q_traj.rows() * 2), self.calc_x)

    def set_trajectory(self,
                       q_init: List[float],
                       X_WG_start: RigidTransform,
                       X_WO: RigidTransform,
                       X_WG_end: RigidTransform,
                       clearance: float = 0.3):
        """
        :param q_init:
        :param X_WG_start:
        :param X_WO:
        :param X_WG_end:
        :param clearance:
        Modifies the current trajectory and hooks it up to the trajectory source output
        """

        p_GO = [0.0, 0.1, 0.]
        R_GO = RotationMatrix.MakeXRotation(np.pi / 2.0).multiply(
            RotationMatrix.MakeZRotation(-np.pi / 2.0))
        X_GO = RigidTransform(R_GO, p_GO)
        X_OG = X_GO.inverse()
        X_WG_grasp = X_WO @ X_OG

        AddMeshcatTriad(self.meshcat, "grasp",
                        length=0.07, radius=0.006, X_PT=X_WG_grasp)

        X_GH = RigidTransform([0, -clearance, 0.])  # H is hover pose
        X_WG_pregrasp = X_WG_grasp @ X_GH

        AddMeshcatTriad(self.meshcat, "end_frame",
                        length=0.15, radius=0.006, X_PT=X_WG_end)

        pose_list = [X_WG_start, X_WG_pregrasp, X_WG_pregrasp, X_WG_grasp, X_WG_pregrasp, X_WG_end]
        # Get the time and number of steps between each pose
        # Create a trajectory for each one and add them all together
        # Create the trajectory using piecewise quaternion slerp(t_knots, pose_knots)
        # is it faster to run slerp on the whole thing, or separate between quaternion and pose?


        # Create a trajectory of quaternions and positions
        quat_traj = []
        pos_traj = []
        for pose in pose_list:


        q_list = []
        for i in range(len(pose_list) - 1):
            if q_list:
                if i == len(pose_list) - 1:
                    init_guess = q_list[-1]
                    init_guess[0] = init_guess[0]
                    q_traj = self.calc_joint_trajectory(pose_list[i], pose_list[i+1], init_guess=init_guess)
                else:
                    q_traj = self.inverse_kinematics(pose_list[i], init_guess=q_list[-1])
            else:
                q_traj = self.inverse_kinematics(pose_list[i], init_guess=np.array([0, 1.57, 0., -1.57, 0., 1.57, 0]))
            q_list += [q_traj]

        self.q_list = q_list
        self.q_traj = self.calc_q_traj()
        self.t_start = 0

    def calc_joint_trajectory(self,
                              X_WE_start: RigidTransform,
                              X_WE_final: RigidTransform,
                              duration: float,
                              frame_E: BodyFrame,  # End-effector frame
                              plant: MultibodyPlant,
                              q_initial_guess: np.ndarray,
                              n_knots: int = 15):
        R_WE_traj = PiecewiseQuaternionSlerp(
            [0, duration], [X_WE_start.rotation().ToQuaternion(),
                            X_WE_final.rotation().ToQuaternion()])
        p_WEo_traj = PiecewisePolynomial.FirstOrderHold(
            [0, duration], np.vstack([X_WE_start.translation(),
                                      X_WE_final.translation()]).T)

        position_tolerance = 0.002
        angle_tolerance = 0.01
        nq = 7

        q_knots = np.zeros((n_knots + 1, nq))
        q_knots[0] = q_initial_guess

        for i in range(1, n_knots + 1):
            t = i / n_knots * duration
            ik = inverse_kinematics.InverseKinematics(plant)
            q_variables = ik.q()

            # Position constraint
            p_WQ_ref = p_WEo_traj.value(t).ravel()
            ik.AddPositionConstraint(
                frameB=frame_E, p_BQ=np.zeros(3),
                frameA=plant.world_frame(),
                p_AQ_lower=p_WQ_ref - position_tolerance,
                p_AQ_upper=p_WQ_ref + position_tolerance)

            # Orientation constraint
            R_WE_ref = RotationMatrix(R_WE_traj.value(t))
            ik.AddOrientationConstraint(
                frameAbar=plant.world_frame(),
                R_AbarA=R_WE_ref,
                frameBbar=frame_E,
                R_BbarB=RotationMatrix(),
                theta_bound=angle_tolerance)

            prog = ik.prog()
            # use the robot posture at the previous knot point as
            # an initial guess.
            prog.SetInitialGuess(q_variables, q_knots[i - 1])
            result = mp.Solve(prog)
            assert result.is_success()
            q_knots[i] = result.GetSolution(q_variables)

        t_knots = np.linspace(0, duration, n_knots + 1)
        q_traj_forward = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            t_knots, q_knots.T, np.zeros(nq), np.zeros(nq))
        q_traj_reverse = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            t_knots, q_knots[::-1].T, np.zeros(nq), np.zeros(nq))

        return q_traj_forward, q_traj_reverse

    def calc_x(self, context, output):
        t = context.get_time() - self.t_start
        q = self.q_traj.value(t).ravel()
        v = self.q_traj.derivative(1).value(t).ravel()
        output.SetFromVector(np.hstack([q, v]))

    def set_t_start(self, t_start_new: float):
        self.t_start = t_start_new

    # Is this method necessary?
    def calc_q_traj(self) -> PiecewisePolynomial:
        """
        Generate a joint configuration trajectory from a beginning and end configuration
        :return: PiecewisePolynomial
        """
        keyframes = [0, 5, 6, 10, 15, 20]

        return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            keyframes, np.vstack(self.q_list).T,
            np.zeros(7), np.zeros(7))


class GripperTrajectorySource(LeafSystem):

    def __init__(self, plant):
        super().__init__()
        self.set_name('pick_and_place_traj')
        self.plant = plant
        self.t_start = 0
        self.q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [0, 3], np.vstack([np.zeros(7), np.zeros(7)]).T,
            np.zeros(7), np.zeros(7))

        self.x_output_port = self.DeclareVectorOutputPort(
            'traj_fingers', BasicVector(2), self.calc_x)

        self.counter = 0

    def calc_x(self, context, output):
        t = context.get_time() - self.t_start
        q = self.q_traj.value(t).ravel()
        v = self.q_traj.derivative(1).value(t).ravel()
        output.SetFromVector(np.hstack([q, v]))


def inverse_kinematics(self, X_WG: RigidTransform, init_guess):
    """
    Given a pose in the world, calculate a reasonable joint configuration for the KUKA iiwa arm that would place
    the gripper in that position.
    :return: Joint configuration for the iiwa
    """
    ik = inverse_kinematics.InverseKinematics(self.plant)
    q_variables = ik.q()

    X_L7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, np.pi / 2.0), [0, 0, 0.114])
    X_WL7 = X_WG @ X_L7G.inverse()

    position_tolerance = 0.005
    frame_L7 = self.plant.GetFrameByName('iiwa_link_7')
    # Position constraint
    p_L7_ref = X_WL7.translation()
    ik.AddPositionConstraint(
        frameB=frame_L7, p_BQ=np.zeros(3),
        frameA=self.plant.world_frame(),
        p_AQ_lower=p_L7_ref - position_tolerance,
        p_AQ_upper=p_L7_ref + position_tolerance)

    # Orientation constraint
    R_WL7_ref = X_WL7.rotation()
    ik.AddOrientationConstraint(
        frameAbar=self.plant.world_frame(),
        R_AbarA=R_WL7_ref,
        frameBbar=frame_L7,
        R_BbarB=RotationMatrix(),
        theta_bound=0.005)

    prog = ik.prog()
    prog.SetInitialGuess(q_variables, init_guess)
    print(init_guess)
    result = Solve(prog)
    # print(prog)
    assert result.is_success()
    print('Success! ', result.GetSolution(q_variables))
    return result.GetSolution(q_variables)


