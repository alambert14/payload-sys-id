import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation
from make_iiwa_and_object import MakeIiwaAndObject

class YeetBot:

    def __init__(self, object_name=None):
        self.diagram = MakeIiwaAndObject(object_name)

        self.viz = self.diagram.GetSubsystemByName('meshcat_visualizer')


if __name__ == '__main__':
    YeetBot('cracker_box')

