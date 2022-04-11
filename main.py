import numpy as np

from pydrake.all import Simulator
from make_iiwa_and_object import MakeIiwaAndObject

class YeetBot:

    def __init__(self, object_name=None):
        self.diagram = MakeIiwaAndObject(object_name)

        self.viz = self.diagram.GetSubsystemByName('meshcat_visualizer')


    def start(self):
        self.viz.reset_recording()
        self.viz.start_recording()

        context = self.diagram.CreateDefaultContext()

        simulator = Simulator(self.diagram)
        self.diagram.Publish(context)
        simulator.AdvanceTo(5.0)




if __name__ == '__main__':
    bot = YeetBot('cracker_box')

    bot.start()

