import numpy as np
from manipulation.meshcat_cpp_utils import AddMeshcatTriad

from pydrake.all import (
    DirectCollocation,
    MathematicalProgram,
    MathematicalProgramResult,
    MultibodyPlant,
    PiecewisePolynomial,
    RigidTransform,
    Solve,
)
from pydrake.math import RotationMatrix
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.multibody import inverse_kinematics


class SinusoidalTrajectorySource(LeafSystem):
    def __init__(self, plant: MultibodyPlant, meshcat, base_frequency, joint_idx = None, timestep = 1e-2, T = 5.0):
        super().__init__()
        self.plant = plant
        self.meshcat = meshcat
        self.T = T
        self.timestep = timestep
        self.freq = base_frequency
        self.joint_idx = joint_idx

        self.q_traj = self.calc_q_traj()

        self.x_output_port = self.DeclareVectorOutputPort(
            'traj_x', BasicVector(self.q_traj.rows() * 2), self.calc_x)
        self.t_start = 0

    def generate_trajectory(self):
        """
        Generate a sinusoidal trajectory
        :return: A list of q vectors and times
        """
        q_list = []
        q_times = []
        for t in np.linspace(0, self.T, num=int(self.T / self.timestep)):
            if not self.joint_idx:
                freqs = np.ones(7) * self.freq
            else:
                freqs = np.zeros(7)
                freqs[self.joint_idx] = self.freq
            q = np.sin(freqs * t)  # Basic sin trajectory, the robot will go crazy
            q_list.append(q)
            q_times.append(t)

        return q_list, q_times

    def calc_x(self, context, output):
        t = context.get_time() - self.t_start
        q = self.q_traj.value(t).ravel()
        v = self.q_traj.derivative(1).value(t).ravel()
        output.SetFromVector(np.hstack([q, v]))

    def set_t_start(self, t_start_new: float):
        self.t_start = t_start_new

    def calc_q_traj(self) -> PiecewisePolynomial:
        """
        Generate a joint configuration trajectory from a beginning and end configuration
        :return: PiecewisePolynomial
        """
        self.q_list, self.q_times = self.generate_trajectory()
        return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            self.q_times, np.vstack(self.q_list).T,
            np.zeros(7), np.zeros(7))


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


## Modified from pangtao/pick-and-place-benchmarking-framework SimpleTrajectory
class PickAndPlaceTrajectorySource(LeafSystem):

    def __init__(self, plant: MultibodyPlant, meshcat,
                 X_L7_start: RigidTransform, X_L7_end: RigidTransform, clearance: float = 0.3):
        super().__init__()
        self.plant = plant
        self.init_guess_start = np.array([0, 1.57, 0., -1.57, 0., 1.57, 0])
        self.init_guess_end = np.array([1.57, 1.57, 0., -1.57, 0., 1.57, 0])
        AddMeshcatTriad(meshcat, "start",
                        length=0.15, radius=0.006, X_PT=X_L7_start)
        AddMeshcatTriad(meshcat, "end",
                        length=0.15, radius=0.006, X_PT=X_L7_end)
        self.start_q = self.inverse_kinematics(X_L7_start, start=True)  # [:-1]
        self.end_q = self.inverse_kinematics(X_L7_end, start=False)  # [:-1]
        self.q_traj = self.calc_q_traj()
        print(self.q_traj.value(3))

        self.x_output_port = self.DeclareVectorOutputPort(
            'traj_x', BasicVector(self.q_traj.rows() * 2), self.calc_x)
        self.t_start = 0

    def inverse_kinematics(self, X_L7: RigidTransform, start=True):
        """
        Given a pose in the world, calculate a reasonable joint configuration for the KUKA iiwa arm that would place
        the end of link 7 in that position.
        :return: Joint configuration for the iiwa
        """
        ik = inverse_kinematics.InverseKinematics(self.plant)
        q_variables = ik.q()

        position_tolerance = 0.01
        frame_L7 = self.plant.GetFrameByName('iiwa_link_7')
        # Position constraint
        p_L7_ref = X_L7.translation()
        ik.AddPositionConstraint(
            frameB=frame_L7, p_BQ=np.zeros(3),
            frameA=self.plant.world_frame(),
            p_AQ_lower=p_L7_ref - position_tolerance,
            p_AQ_upper=p_L7_ref + position_tolerance)

        # Orientation constraint
        R_WL7_ref = X_L7.rotation()  # RotationMatrix(R_WE_traj.value(t))
        ik.AddOrientationConstraint(
            frameAbar=self.plant.world_frame(),
            R_AbarA=R_WL7_ref,
            frameBbar=frame_L7,
            R_BbarB=RotationMatrix(),
            theta_bound=0.01)

        prog = ik.prog()
        # use the robot posture at the previous knot point as
        # an initial guess.
        if start:
            init_guess = self.init_guess_start
        else:
            init_guess = self.init_guess_end
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

    def calc_q_traj(self) -> PiecewisePolynomial:
        """
        Generate a joint configuration trajectory from a beginning and end configuration
        :return: PiecewisePolynomial
        """
        return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            [0, 3], np.vstack([self.start_q, self.end_q]).T,
            np.zeros(7), np.zeros(7))










