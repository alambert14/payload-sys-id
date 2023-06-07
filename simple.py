import numpy as np

from pydrake.all import Simulator

from make_iiwa_and_object import MakeIiwaAndObject
from utils import calc_data_matrix, plot_all_parameters_est

from pydrake.all import (Adder, AddMultibodyPlantSceneGraph, Demultiplexer,
                         DiagramBuilder, InverseDynamicsController, FindResourceOrThrow,
                         MakeMultibodyStateToWsgStateSystem,
                         MeshcatVisualizerCpp, MeshcatAnimation, MultibodyPlant, Parser,
                         PassThrough, PrismaticJoint, PiecewisePolynomial, Polynomial, RigidTransform,
                         SchunkWsgPositionController,
                         StateInterpolatorWithDiscreteDerivative, JointSliders, Solve)
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.systems.primitives import LogVectorOutput
from pydrake.multibody.tree import RevoluteJoint, FixedOffsetFrame
from manipulation.meshcat_cpp_utils import StartMeshcat


class SysIDSim():

    def __init__(self):
        super().__init__(self)
        self.DOF = 7
        # self.q0 = [0.0, 0.1, 0, -1.2, 0, 1.6, 0]
        self.q0 = [0.0] * 7
        self.diagram, self.plant, self.meshcat, self.state_logger, self.torque_logger, self.viz = self.make_env()
        self.simulator = Simulator(self.diagram)
        print('created diagram')

    def render_system_with_graphviz(self, output_file="system_view.gz"):
        """ Renders the Drake system (presumably a diagram,
        otherwise this graph will be fairly trivial) using
        graphviz to a specified file. Borrowed from @pang-tao"""
        from graphviz import Source
        string = self.diagram.GetGraphvizString()
        src = Source(string)
        src.render(output_file, view=False)

    def make_env(self):
        """
        Create the iiwa and welded object system diagram
        :param object_name: Name of the object model to add. If none, just create the iiwa.
        :param DOF: Degrees of freedom of the iiwa (currently supports 7 and 2)
        :param time_step:
        :return:
        """
        builder = DiagramBuilder()

        plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                         time_step=0)
        meshcat = StartMeshcat()
        animation = MeshcatAnimation()
        meshcat.SetAnimation(animation)

        # Add iiwa
        parser = Parser(plant)
        AddPackagePaths(parser)  # Russ's manipulation repo.
        add_package_paths_local(parser)  # local.

        ProcessModelDirectives(LoadModelDirectives('models/workstation.yaml'), plant, parser)

        iiwa = plant.GetModelInstanceByName('iiwa')  # GetBodyByName('iiwa')

        # Set default positions:
        index = 0
        for joint_index in plant.GetJointIndices(iiwa):
            joint = plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(self.q0[index])
                index += 1

        parser = Parser(plant)
        obj_idx = parser.AddModelFromFile(
            'models/006_mustard_bottle.sdf', 'mustard_bottle')

        X_7G = RigidTransform([0, 0.114, 0])
        X_7O = RigidTransform(RollPitchYaw([np.pi / 2, 0, 0]), [0, 0.1, 0])

        joint_offset = FixedOffsetFrame(
            'offset',
            plant.GetFrameByName('iiwa_link_7'),  # parent frame P
            X_7G,  # X_PF
        )
        plant.AddFrame(joint_offset)
        plant.Finalize()

        num_iiwa_positions = plant.num_positions(iiwa)

        # I need a PassThrough system so that I can export the input port.
        iiwa_position = builder.AddSystem(PassThrough(self.DOF * 2))
        builder.ExportOutput(iiwa_position.get_output_port(),
                             "iiwa_position_command")

        # Export the iiwa "state" outputs.
        demux = builder.AddSystem(
            Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions))
        builder.Connect(plant.get_state_output_port(iiwa), demux.get_input_port())
        builder.ExportOutput(demux.get_output_port(0), "iiwa_position_measured")
        builder.ExportOutput(demux.get_output_port(1), "iiwa_velocity_estimated")
        builder.ExportOutput(plant.get_state_output_port(iiwa),
                             "iiwa_state_estimated")

        # Make the plant for the iiwa controller to use.
        controller_plant = MultibodyPlant(time_step=0)
        controller_iiwa = AddIiwa(controller_plant, self.DOF)

        controller_plant.Finalize()

        # Create sample trajectory, ask Lirui which one would be best
        X_L7_start = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 0)), [0.6, 0., 0.6])
        X_L7_end = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 3.14)), [-0.4, -0.3, 0.6])
        # q_source = builder.AddSystem(PickAndPlaceTrajectorySource(controller_plant, meshcat, X_L7_start, X_L7_end))
        q_source = builder.AddSystem(
            SinusoidalTrajectorySource(controller_plant, meshcat, self.DOF, base_frequency=1, joint_idx=6, T=10.))

        iiwa_controller = builder.AddSystem(
            InverseDynamicsController(controller_plant,
                                      kp=[500, 500, 500, 500, 500, 5000, 5000],
                                      ki=[1] * self.DOF,
                                      kd=[200] * self.DOF,  # 200
                                      has_reference_acceleration=False))

        iiwa_controller.set_name("iiwa_controller")
        builder.Connect(plant.get_state_output_port(iiwa),
                        iiwa_controller.get_input_port_estimated_state())

        # Add in the feed-forward torque
        adder = builder.AddSystem(Adder(2, self.DOF))
        builder.Connect(iiwa_controller.get_output_port_control(),
                        adder.get_input_port(0))
        # Use a PassThrough to make the port optional (it will provide zero values
        # if not connected).
        torque_passthrough = builder.AddSystem(PassThrough([0] * self.DOF))
        builder.Connect(torque_passthrough.get_output_port(),
                        adder.get_input_port(1))
        builder.ExportInput(torque_passthrough.get_input_port(),
                            "iiwa_feedforward_torque")
        builder.Connect(adder.get_output_port(),
                        plant.get_actuation_input_port(iiwa))

        builder.Connect(iiwa_position.get_output_port(),
                        iiwa_controller.get_input_port_desired_state())

        # Export commanded torques.
        builder.ExportOutput(adder.get_output_port(), "iiwa_torque_commanded")
        builder.ExportOutput(adder.get_output_port(), "iiwa_torque_measured")

        builder.ExportOutput(plant.get_generalized_contact_forces_output_port(iiwa),
                             "iiwa_torque_external")

        # Export "cheat" ports.
        builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
        builder.ExportOutput(plant.get_contact_results_output_port(),
                             "contact_results")
        builder.ExportOutput(plant.get_state_output_port(),
                             "plant_continuous_state")
        builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

        viz = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

        # Attach trajectory
        builder.Connect(q_source.get_output_port(),
                        iiwa_position.get_input_port())
        # builder.Connect(wsg_traj_source.get_output_port(),
        #                 wsg_controller.get_desired_position_input_port())

        state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
        torque_logger = LogVectorOutput(adder.get_output_port(), builder)
        diagram = builder.Build()
        diagram.set_name("ManipulationStation")

        return diagram, plant, meshcat, state_logger, torque_logger, viz

    def start(self):
        self.render_system_with_graphviz()
        self.viz.StartRecording()

        context = self.diagram.CreateDefaultContext()
        context_plant = self.plant.GetMyContextFromRoot(context)

        self.diagram.Publish(context)
        self.simulator.AdvanceTo(20)

        self.viz.StopRecording()
        self.viz.PublishRecording()

        html = self.meshcat.StaticHtml()
        with open(f"oscillatory.html", "w") as f:
            f.write(html)

        body = self.plant.GetBodyByName('base_link_mustard')




        state_log = self.state_logger.FindLog(self.simulator.get_context())
        torque_log = self.torque_logger.FindLog(self.simulator.get_context())

        all_alpha = calc_data_matrix(self.plant, state_log, torque_log, self.DOF)

        com = body.CalcCenterOfMassInBodyFrame(context)
        inertia = body.CalcSpatialInertiaInBodyFrame(context).CopyToFullMatrix6()[:3,:3]
        m = 0.603
        ground_truth = [
            m,
            m * com[0], m * com[1], m * com[2],
            inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2],
        ]

        plot_all_parameters_est(all_alpha, ground_truth)

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

class SinusoidalTrajectorySource(LeafSystem):
    def __init__(self, plant: MultibodyPlant, meshcat, DOF, base_frequency, joint_idx = None, timestep = 1e-2, T = 5.0):
        super().__init__()
        self.plant = plant
        self.meshcat = meshcat
        self.T = T
        self.DOF = DOF
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
                freqs = np.ones(self.DOF) * self.freq
            else:
                freqs = np.zeros(self.DOF)
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
            np.zeros(self.DOF), np.zeros(self.DOF))


if __name__ == '__main__':
    bot = SysIDSim()
    bot.start()

