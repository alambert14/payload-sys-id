import numpy as np

from pydrake.all import Simulator
from make_iiwa_and_object import MakeIiwaAndObject
from utils import calc_data_matrix, plot_all_parameters_est
from pcl_to_inertia import calculate_ground_truth_parameters

class YeetBot:

    def __init__(self, object_name=None):
        self.diagram, self.plant, self.meshcat, self.state_logger, self.torque_logger = MakeIiwaAndObject(object_name)
        print('created diagram')
        # self.viz = self.diagram.GetSubsystemByName('meshcat_visualizer')


    def start(self):
        self.render_system_with_graphviz()
        # self.viz.reset_recording()
        # self.viz.start_recording()

        context = self.diagram.CreateDefaultContext()

        simulator = Simulator(self.diagram)
        # integrator = simulator.get_mutable_integrator()
        # integrator.set_fixed_step_mode(True)
        self.diagram.Publish(context)
        simulator.AdvanceTo(50.0)

        body = self.plant.GetBodyByName('base_link_mustard')
        print('Mustard CoM: ', body.CalcCenterOfMassInBodyFrame(context))
        print('Mustard Inertia: ', body.CalcSpatialInertiaInBodyFrame(context).CopyToFullMatrix6()[:3,:3])

        state_log = self.state_logger.FindLog(simulator.get_context())
        torque_log = self.torque_logger.FindLog(simulator.get_context())

        # object_mass = calc_mass(self.plant, state_log, torque_log)
        # rint('calculated_mass: ', object_mass)
        all_alpha = calc_data_matrix(self.plant, state_log, torque_log)
        try:
            all_data = np.loadtxt('total_data.txt')
            print('found data file')
            all_data = np.vstack((all_data, all_alpha[-1]))
            np.savetxt('total_data.txt', all_data)
        except OSError:
            print('creating data file')
            all_data = all_alpha[-1]
            np.savetxt('total_data.txt', all_data)

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


    def render_system_with_graphviz(self, output_file="system_view.gz"):
        """ Renders the Drake system (presumably a diagram,
        otherwise this graph will be fairly trivial) using
        graphviz to a specified file. Borrowed from @pang-tao"""
        from graphviz import Source
        string = self.diagram.GetGraphvizString()
        src = Source(string)
        src.render(output_file, view=False)


if __name__ == '__main__':
    bot = YeetBot('mustard_bottle')

    bot.start()

