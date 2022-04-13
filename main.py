import numpy as np

from pydrake.all import Simulator
from make_iiwa_and_object import MakeIiwaAndObject

class YeetBot:

    def __init__(self, object_name=None):
        self.diagram = MakeIiwaAndObject(object_name)

        self.viz = self.diagram.GetSubsystemByName('meshcat_visualizer')


    def start(self):
        self.render_system_with_graphviz()
        self.viz.reset_recording()
        self.viz.start_recording()

        context = self.diagram.CreateDefaultContext()

        simulator = Simulator(self.diagram)
        self.diagram.Publish(context)
        simulator.AdvanceTo(5.0)

    def render_system_with_graphviz(self, output_file="system_view.gz"):
        """ Renders the Drake system (presumably a diagram,
        otherwise this graph will be fairly trivial) using
        graphviz to a specified file. Borrowed from @pang-tao"""
        from graphviz import Source
        string = self.diagram.GetGraphvizString()
        src = Source(string)
        src.render(output_file, view=False)


if __name__ == '__main__':
    bot = YeetBot('cracker_box')

    bot.start()

