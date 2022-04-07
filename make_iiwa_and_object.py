####################################################################################
# Modified from the ManipulationStation example in pydrake:
# https://github.com/RussTedrake/manipulation/blob/master/manipulation_station.ipynb
####################################################################################

from IPython.display import display, SVG
import matplotlib.pyplot as plt
import numpy as np
import os
import pydot
import sys

from pydrake.all import (Adder, AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, Demultiplexer,
                         DiagramBuilder, InverseDynamicsController, FindResourceOrThrow,
                         MakeMultibodyStateToWsgStateSystem,
                         MeshcatVisualizerCpp, MultibodyPlant, Parser,
                         PassThrough, RigidTransform, RollPitchYaw,
                         SchunkWsgPositionController,
                         StateInterpolatorWithDiscreteDerivative)
from manipulation.meshcat_cpp_utils import StartMeshcat
from manipulation.scenarios import AddCameraBox, AddIiwa, AddWsg, AddRgbdSensors
from manipulation.utils import FindResource
from manipulation import running_as_notebook

from models import object_library


def MakeIiwaAndObject(object_name=None, time_step=0.002):
    """
    Create the iiwa and welded object system diagram
    :param object_name: Name of the object model to add. If none, just create the iiwa.
    :param time_step:
    :return:
    """
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step)
    iiwa = AddIiwa(plant)
    if object is not None:
        object = AddObject(plant, iiwa, object_name)
    plant.Finalize()

    num_iiwa_positions = plant.num_positions(iiwa)

    # I need a PassThrough system so that I can export the input port.
    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
    builder.ExportInput(iiwa_position.get_input_port(), "iiwa_position")
    builder.ExportOutput(iiwa_position.get_output_port(), "iiwa_position_command")

    # Export the iiwa "state" outputs.
    demux = builder.AddSystem(Demultiplexer(
        2 * num_iiwa_positions, num_iiwa_positions))
    builder.Connect(plant.get_state_output_port(iiwa), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "iiwa_position_measured")
    builder.ExportOutput(demux.get_output_port(1), "iiwa_velocity_estimated")
    builder.ExportOutput(plant.gte_state_output_port(iiwa), "iiwa_state_estimated")

    # Make the plant for the iiwa controller to use.
    controller_plant = MultibodyPlant(time_step=time_step)
    controller_iiwa = AddIiwa(controller_plant)
    if object_name is not None:
        AddObject(controller_plant, controller_iiwa, object_name)
    controller_plant.Finalize()

    # Add the iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
            kp=[100]*num_iiwa_positions,
            ki=[1]*num_iiwa_positions,
            kd=[20]*num_iiwa_positions,
            has_reference_acceleration=False))
    iiwa_controller.set_name("iiwa_controller")
    builder.Connect(
        plant.get_state_output_port(iiwa), iiwa_controller.get_input_port_estimated_state())

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values if not connected).
    torque_passthrough = builder.AddSystem(PassThrough([0]*num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(),
                        "iiwa_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(iiwa))

    # Add discrete derivative to command velocities.
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_iiwa_positions, time_step, suppress_initial_transient=True))
    desired_state_from_position.set_name("desired_state_from_position")
    builder.Connect(desired_state_from_position.get_output_port(),
                    iiwa_controller.get_input_port_desired_state())
    builder.Connect(iiwa_position.get_output_port(),
                    desired_state_from_position.get_input_port())

    # Export commanded torques.
    builder.ExportOutput(adder.get_output_port(), "iiwa_torque_commanded")
    builder.ExportOutput(adder.get_output_port(), "iiwa_torque_measured")

    builder.ExportOutput(plant.get_generalized_contact_forces_output_port(iiwa),
                         "iiwa_torque_external")

    # Wsg controller.
    # wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    # wsg_controller.set_name("wsg_controller")
    # builder.Connect(
    #     wsg_controller.get_generalized_force_output_port(),
    #     plant.get_actuation_input_port(wsg))
    # builder.Connect(plant.get_state_output_port(wsg),
    #                 wsg_controller.get_state_input_port())
    # builder.ExportInput(wsg_controller.get_desired_position_input_port(),
    #                     "wsg_position")
    # builder.ExportInput(wsg_controller.get_force_limit_input_port(),
    #                     "wsg_force_limit")
    # wsg_mbp_state_to_wsg_state = builder.AddSystem(
    #     MakeMultibodyStateToWsgStateSystem())
    # builder.Connect(plant.get_state_output_port(wsg),
    #                 wsg_mbp_state_to_wsg_state.get_input_port())
    # builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(),
    #                      "wsg_state_measured")
    # builder.ExportOutput(wsg_controller.get_grip_force_output_port(),
    #                      "wsg_force_measured")

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                         "plant_continuous_state")

    viz = ConnectMeshcatVisualizer(
        builder, scene_graph, zmq_url='tcp://127.0.0.1:6000', prefix="environment")


    diagram = builder.Build()
    return diagram


####################################################################################
# Modified from the AddWSG example in pydrake:
# https://github.com/RussTedrake/manipulation/blob/master/manipulation/scenarios.py
####################################################################################
def AddObject(plant, iiwa_model_instance, object_name, roll=np.pi / 2.0):
    """
    Add an object welded to the 7th link of the iiwa
    :param plant:
    :param iiwa_model_instance:
    :param object_name:
    :param roll:
    :return:
    """
    parser = Parser(plant)
    try:
        object = parser.AddModelFromFile(
            FindResource(object_libary[object_name]), object_name)
    except KeyError:
        raise KeyError(f'Cannot find {object_name} in the object library.')

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, roll), [0, 0, 0.114])
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa_model_instance),
                     plant.GetFrameByName("body", object), X_7G)
    return object