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
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.multibody import inverse_kinematics


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

    def __init__(self, plant: MultibodyPlant): # ,
                 # q_init: List[float],
                 # X_WG_start: RigidTransform,
                 # X_WO: RigidTransform,
                 # X_WG_end: RigidTransform, clearance: float = 0.3):
        super().__init__()
        self.set_name('pick_and_place_traj')
        self.plant = plant
        self.t_start = 0
        self.q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [0, 3], np.vstack([np.zeros(7), np.zeros(7)]).T,
            np.zeros(7), np.zeros(7))

        self.x_output_port = self.DeclareVectorOutputPort(
            'traj_x', BasicVector(self.q_traj.rows() * 2), self.calc_x)

        # self.set_trajectory(q_init, X_WG_start, X_WO, X_WG_end, clearance)

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

        p_GO = [0, 0.11, 0]
        R_GO = RotationMatrix.MakeXRotation(np.pi / 2.0).multiply(
            RotationMatrix.MakeZRotation(np.pi / 2.0))
        X_GO = RigidTransform(R_GO, p_GO)
        X_OG = X_GO.inverse()
        X_WG_grasp = X_WO @ X_OG

        X_GH = RigidTransform([0, -0.08, 0])  # H is hover pose
        X_WG_pregrasp = X_WG_grasp @ X_GH


        pose_list = [X_WG_start, X_WG_pregrasp, X_WG_grasp, X_WG_pregrasp, X_WG_end]

        q_list = []
        for pose in pose_list:
            if q_list:
                q = self.inverse_kinematics(pose, init_guess=q_list[-1][:8])
            else:
                q = self.inverse_kinematics(pose, init_guess=np.array([0, 1.57, 0., -1.57, 0., 1.57, 0, 0]))

            q_list.append(q)

        self.q_list = q_list
        self.q_traj = self.calc_q_traj()
        self.t_start = 0

    def inverse_kinematics(self, X_WG: RigidTransform, init_guess):
        """
        Given a pose in the world, calculate a reasonable joint configuration for the KUKA iiwa arm that would place
        the gripper in that position.
        :return: Joint configuration for the iiwa
        """
        ik = inverse_kinematics.InverseKinematics(self.plant)
        q_variables = ik.q()
        print(len(q_variables))

        X_L7G = RigidTransform([0, -0.11, 0])  # Check that this is correct
        X_WL7 = X_WG @ X_L7G

        position_tolerance = 0.01
        frame_L7 = self.plant.GetFrameByName('iiwa_link_7')
        # Position constraint
        p_L7_ref = X_WL7.translation()
        ik.AddPositionConstraint(
            frameB=frame_L7, p_BQ=np.zeros(3),
            frameA=self.plant.world_frame(),
            p_AQ_lower=p_L7_ref - position_tolerance,
            p_AQ_upper=p_L7_ref + position_tolerance)

        # Orientation constraint
        R_WL7_ref = X_WG.rotation()  # RotationMatrix(R_WE_traj.value(t))
        ik.AddOrientationConstraint(
            frameAbar=self.plant.world_frame(),
            R_AbarA=R_WL7_ref,
            frameBbar=frame_L7,
            R_BbarB=RotationMatrix(),
            theta_bound=0.01)

        prog = ik.prog()
        print(len(q_variables), len(init_guess))
        prog.SetInitialGuess(q_variables, init_guess)
        print(prog)
        result = Solve(prog)
        assert result.is_success()
        return result.GetSolution(q_variables)

    def calc_x(self, context, output):
        t = context.get_time() - self.t_start
        q = self.q_traj.value(t).ravel()
        v = self.q_traj.derivative(1).value(t).ravel()
        output.SetFromVector(np.hstack([q, v]))

    def set_t_start(self, t_start_new: float):
        self.t_start = t_start_new

    # Is this method necessary?
    def calc_q_traj(self, q_list) -> PiecewisePolynomial:
        """
        Generate a joint configuration trajectory from a beginning and end configuration
        :return: PiecewisePolynomial
        """

        return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [0, 5], np.vstack(q_list).T,
            np.zeros(7), np.zeros(7))
