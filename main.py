import numpy as np

from pydrake.all import Simulator
from make_iiwa_and_object import MakeIiwaAndObject
from utils import calc_mass, calc_data_matrix

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
        simulator.AdvanceTo(3.0)

        state_log = self.state_logger.FindLog(simulator.get_context())
        torque_log = self.torque_logger.FindLog(simulator.get_context())

        # object_mass = calc_mass(self.plant, state_log, torque_log)
        # rint('calculated_mass: ', object_mass)
        print(calc_data_matrix(self.plant, state_log, torque_log)) #, mass=0.603000))# 0.63282601)) # object_mass))


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

