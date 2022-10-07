import os
import pydrake
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder,
    Parser, ProcessModelDirectives, LoadModelDirectives, ContactModel,
    Simulator, MeshcatVisualizerCpp,
)

from manipulation.scenarios import AddRgbdSensors
from manipulation.utils import AddPackagePaths
from manipulation.meshcat_cpp_utils import StartMeshcat, AddMeshcatTriad

def add_package_paths_local(parser: Parser):
    parser.package_map().Add(
        "drake_manipulation_models",
        os.path.join(pydrake.common.GetDrakePath(),
                     "manipulation/models"))

    # parser.package_map().Add('iiwa_controller',
    #                          iiwa_controller_models_dir)

    parser.package_map().Add("local", 'models')
    parser.package_map().PopulateFromFolder('models')


def test_directives(filename):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=2e-4)
    parser = Parser(plant)
    AddPackagePaths(parser)  # Russ's manipulation repo.
    add_package_paths_local(parser)  # local.
    ProcessModelDirectives(LoadModelDirectives(filename), plant, parser)

    # Set contact model.
    plant.set_contact_model(ContactModel.kPointContactOnly)
    plant.Finalize()
    AddRgbdSensors(builder, plant, scene_graph)

    meshcat = StartMeshcat()
    MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
    # viz = ConnectMeshcatVisualizer(
    #     builder, scene_graph, zmq_url="tcp://127.0.0.1:7000", prefix="environment")
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    # viz.load()
    diagram.Publish(context)

    simulator = Simulator(diagram, context)

    simulator.AdvanceTo(5.0)

    # viz.reset_recording()
    # viz.start_recording()
    #
    # viz.stop_recording()
    # viz.publish_recording()

if __name__ == '__main__':
    test_directives('models/workstation.yaml')