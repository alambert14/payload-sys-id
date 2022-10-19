import numpy as np
from manipulation.meshcat_cpp_utils import AddMeshcatTriad

from pydrake.all import Simulator
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw

from make_iiwa_and_object import MakeIiwaAndObject, MakePlaceBot
from utils import calc_data_matrix, plot_all_parameters_est, detect_slip
from pcl_to_inertia import calculate_ground_truth_parameters


class Bot:

    def __init__(self, object_name=None):
        pass

    def render_system_with_graphviz(self, output_file="system_view.gz"):
        """ Renders the Drake system (presumably a diagram,
        otherwise this graph will be fairly trivial) using
        graphviz to a specified file. Borrowed from @pang-tao"""
        from graphviz import Source
        string = self.diagram.GetGraphvizString()
        src = Source(string)
        src.render(output_file, view=False)

    def start(self):
        raise NotImplementedError


class YeetBot(Bot):

    def __init__(self, object_name=None):
        super().__init__(self)
        self.diagram, self.plant, self.meshcat, self.state_logger, self.torque_logger = MakeIiwaAndObject(object_name)
        self.simulator = Simulator(self.diagram)
        print('created diagram')

    def start(self):
        self.render_system_with_graphviz()
        # self.viz.reset_recording()
        # self.viz.start_recording()

        context = self.diagram.CreateDefaultContext()
        context_plant = self.plant.GetMyContextFromRoot(context)
        # integrator = simulator.get_mutable_integrator()
        # integrator.set_fixed_step_mode(True)
        self.diagram.Publish(context)
        self.simulator.AdvanceTo(0.0)

        body = self.plant.GetBodyByName('base_link_mustard')
        mustard_frame = self.plant.GetFrameByName('base_link_mustard')
        AddMeshcatTriad(self.meshcat, "mustard_pose",
                        length=0.15, radius=0.006, X_PT=mustard_frame.CalcPoseInWorld(context_plant))
        print('Mustard CoM: ', body.CalcCenterOfMassInBodyFrame(context))
        print('Mustard Inertia: ', body.CalcSpatialInertiaInBodyFrame(context).CopyToFullMatrix6()[:3,:3])

        state_log = self.state_logger.FindLog(self.simulator.get_context())
        torque_log = self.torque_logger.FindLog(self.simulator.get_context())

        # object_mass = calc_mass(self.plant, state_log, torque_log)
        # rint('calculated_mass: ', object_mass)
        all_alpha = calc_data_matrix(self.plant, state_log, torque_log)
        # try:
        #     all_data = np.loadtxt('total_data.txt')
        #     print('found data file')
        #     all_data = np.vstack((all_data, all_alpha[-1]))
        #     np.savetxt('total_data.txt', all_data)
        # except OSError:
        #     print('creating data file')
        #     all_data = all_alpha[-1]
        #     np.savetxt('total_data.txt', all_data)

        # np.savetxt('all_alpha.txt', all_alpha)
        com = body.CalcCenterOfMassInBodyFrame(context)
        inertia = body.CalcSpatialInertiaInBodyFrame(context).CopyToFullMatrix6()[:3,:3]
        ground_truth = [
            0.603,
            com[0], com[1], com[2],
            inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2],
        ]

        # calculate_ground_truth_parameters('nontextured.ply')


        plot_all_parameters_est(all_alpha, ground_truth)


class PlaceBot(Bot):

    def __init__(self, object_name=None):
        super().__init__(self)
        self.diagram, self.plant, self.meshcat, self.state_logger, self.torque_logger, self.object = MakePlaceBot(object_name)
        self.simulator = Simulator(self.diagram)

    def start(self):
        self.render_system_with_graphviz()
        # self.viz.reset_recording()
        # self.viz.start_recording()

        context = self.diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyContextFromRoot(context)

        X_WO = RigidTransform(RotationMatrix(), [0.5, 0., 0.05])
        # indices = self.plant.GetBodyIndices(self.object)
        # obj = self.plant.get_body(indices[0])
        self.plant.SetFreeBodyPose(plant_context,
                                   self.object,
                                   X_WO)

        print('setting free body pose')
        # integrator = simulator.get_mutable_integrator()
        # integrator.set_fixed_step_mode(True)

        self.diagram.Publish(context)
        self.simulator = Simulator(self.diagram, context)
        self.simulator.AdvanceTo(3)
        self.simulator.set_target_realtime_rate(0.)


        X_WO = self.plant.EvalBodyPoseInWorld(plant_context, self.object)
        print('X_WO: ', X_WO)

        # AddMeshcatTriad(meshcat, "grasp",
        #                 length=0.07, radius=0.006, X_PT=X_WG_grasp)

        q_init = [0, 1.57, 0., -1.57, 0., 1.57, 0, 0]

        R_WG_start = RotationMatrix.MakeXRotation(np.pi / 2.0).multiply(
            RotationMatrix.MakeZRotation(-np.pi / 2.0))
        X_WG_start = RigidTransform(R_WG_start, [0.6, 0., 0.6])

        R_WG_end = RotationMatrix.MakeXRotation(np.pi / 2.0).multiply(
            RotationMatrix.MakeZRotation(np.pi))
        X_WG_end = RigidTransform(R_WG_end, [0, 0.5, 0.5])

        traj_source = self.diagram.GetSubsystemByName('pick_and_place_traj')
        traj_source.set_trajectory(q_init, X_WG_start, X_WO, X_WG_end)

        self.simulator.AdvanceTo(20)

        state_log = self.state_logger.FindLog(self.simulator.get_context())
        torque_log = self.torque_logger.FindLog(self.simulator.get_context())

        # detect_slip(self.plant, state_log)


        # object_mass = calc_mass(self.plant, state_log, torque_log)
        # rint('calculated_mass: ', object_mass)
        # all_alpha = calc_data_matrix(self.plant, state_log, torque_log)

        # np.savetxt('all_alpha.txt', all_alpha)
        # ground_truth = calculate_ground_truth_parameters('nontextured.ply')
        # plot_all_parameters_est(all_alpha, ground_truth)


if __name__ == '__main__':
    bot = YeetBot('mustard_bottle')

    bot.start()

