####################################################################################
# Modified from the ManipulationStation example in pydrake:
# https://github.com/RussTedrake/manipulation/blob/master/manipulation_station.ipynb
####################################################################################

from IPython.display import display, SVG, Math
import matplotlib.pyplot as plt
import numpy as np
import os
import pydot
import sys

from pydrake.all import (Adder, AddMultibodyPlantSceneGraph, Demultiplexer,
                         DiagramBuilder, InverseDynamicsController, FindResourceOrThrow,
                         MakeMultibodyStateToWsgStateSystem,
                         MeshcatVisualizerCpp, MultibodyPlant, Parser,
                         PassThrough, PrismaticJoint, Polynomial, RigidTransform,
                         SchunkWsgPositionController,
                         StateInterpolatorWithDiscreteDerivative, ToLatex, JointSliders)
from manipulation.meshcat_cpp_utils import StartMeshcat, AddMeshcatTriad
from manipulation.scenarios import AddCameraBox, AddIiwa, AddWsg, AddRgbdSensors
from manipulation.utils import FindResource
from manipulation import running_as_notebook
from graphviz import Source
from pydrake.geometry.render import MakeRenderEngineVtk, RenderEngineVtkParams, DepthRenderCamera, RenderCameraCore, \
    ClippingRange, DepthRange
from pydrake.multibody.tree import RevoluteJoint, FixedOffsetFrame, SpatialInertia_, RotationalInertia, \
    RotationalInertia_
from pydrake.symbolic import Expression, MakeVectorVariable, MakeMatrixVariable, Variable, DecomposeLumpedParameters
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.geometry import Meshcat
from pydrake.systems.primitives import TrajectorySource, LogVectorOutput
from pydrake.systems.sensors import CameraInfo, RgbdSensor
from pydrake.trajectories import PiecewisePolynomial

from sysid_trajectory import PickAndPlaceTrajectorySource, SimpleTrajectorySource
from utils import remove_terms_with_small_coefficients

from models.object_library import object_library


def MakeIiwaAndObject(object_name=None, time_step=0):
    """
    Create the iiwa and welded object system diagram
    :param object_name: Name of the object model to add. If none, just create the iiwa.
    :param time_step:
    :return:
    """
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=time_step)
    meshcat = StartMeshcat()
    iiwa = AddIiwa(plant)
    # wsg = AddWsg(plant, iiwa)
    # if plant_setup_callback:
    #     plant_setup_callback(plant)
    obj_idx = AddObject(plant, iiwa, object_name)
    print('added object')

    print(plant.num_joints())

    plant.Finalize()
    print('finalized plant')

    num_iiwa_positions = plant.num_positions(iiwa)
    num_iiwa_velocities = plant.num_positions(iiwa)
    print(num_iiwa_positions)

    # I need a PassThrough system so that I can export the input port.
    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions + num_iiwa_velocities))
    # builder.ExportInput(iiwa_position.get_input_port(), "iiwa_position")
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
    controller_plant = MultibodyPlant(time_step=time_step)
    controller_iiwa = AddIiwa(controller_plant)
    # AddWsg(controller_plant, controller_iiwa, welded=True)

    controller_plant.Finalize()

    # Create sample trajectory
    # q_knots = np.array([[1.57, 0., 0., -1.57, 0., 1.57, 0,
    #                      0, 0, 0, 0, 0, 0, 0],
    #                     [1.57, 0., 0., -1.57, 0., 1.57, 0,
    #                      0, 0, 0, 0, 0, 0, 0]
    # ])
    # traj = PiecewisePolynomial.ZeroOrderHold([0, 1], q_knots.T)
    X_L7_start = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 0)), [0.6, 0., 0.6])
    X_L7_end = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 0.)), [-0.4, -0.3, 0.6])
    q_source = builder.AddSystem(PickAndPlaceTrajectorySource(controller_plant, X_L7_start, X_L7_end))
    AddMeshcatTriad(meshcat, "start_frame",
                    length=0.15, radius=0.006, X_PT=X_L7_start)
    AddMeshcatTriad(meshcat, "end_frame",
                    length=0.15, radius=0.006, X_PT=X_L7_end)

    # Add the iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(controller_plant,
                                  kp=[100, 100, 100, 100, 100, 100, 10000],
                                  ki=[1] * num_iiwa_positions,
                                  kd=[20] * num_iiwa_positions,
                                  has_reference_acceleration=False))
    iiwa_controller.set_name("iiwa_controller")
    builder.Connect(plant.get_state_output_port(iiwa),
                    iiwa_controller.get_input_port_estimated_state())

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values
    # if not connected).
    torque_passthrough = builder.AddSystem(PassThrough([0]
                                                       * num_iiwa_positions))
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

    MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

    # Attach trajectory
    builder.Connect(q_source.get_output_port(),
                    iiwa_position.get_input_port())

    state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    torque_logger = LogVectorOutput(adder.get_output_port(), builder)
    diagram = builder.Build()
    diagram.set_name("ManipulationStation")

    string = diagram.GetGraphvizString()
    src = Source(string)
    src.render('graph.gz', view=False)

    return diagram, plant, meshcat, state_logger, torque_logger


def MakePlaceBot(object_name = None, time_step = 2e-4):
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=time_step)
    meshcat = StartMeshcat()
    iiwa = AddIiwa(plant)
    wsg = AddWsg(plant, iiwa)
    # if plant_setup_callback:
    #     plant_setup_callback(plant)
    obj_idx = AddGraspedObject(plant, wsg, object_name)
    AddTable(plant, iiwa)
    camera = AddRgbdSensor(builder, scene_graph, RigidTransform([0.5, 0., 0.5]))

    plant.Finalize()
    print('finalized plant')

    num_iiwa_positions = plant.num_positions(iiwa)
    num_iiwa_velocities = plant.num_positions(iiwa)
    print(num_iiwa_positions)

    # I need a PassThrough system so that I can export the input port.
    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions + num_iiwa_velocities))
    # builder.ExportInput(iiwa_position.get_input_port(), "iiwa_position")
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
    controller_plant = MultibodyPlant(time_step=time_step)
    controller_iiwa = AddIiwa(controller_plant)
    AddWsg(controller_plant, controller_iiwa, welded=True)
    # Controller was bugging out before because it had unexpected forces from the gripper?

    controller_plant.Finalize()

    # Create sample trajectory
    # q_knots = np.array([[1.57, 0., 0., -1.57, 0., 1.57, 0,
    #                      0, 0, 0, 0, 0, 0, 0],
    #                     [1.57, 0., 0., -1.57, 0., 1.57, 0,
    #                      0, 0, 0, 0, 0, 0, 0]])
    # traj = PiecewisePolynomial.ZeroOrderHold([0, 1], q_knots.T)
    X_L7_start = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 0)), [0.6, 0., 0.6])
    X_L7_end = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 1.57)), [-0.4, -0.3, 0.6])
    q_source = builder.AddSystem(PickAndPlaceTrajectorySource(controller_plant, meshcat))
    # AddMeshcatTriad(meshcat, "start_frame",
    #                 length=0.15, radius=0.006, X_PT=X_L7_start)
    # AddMeshcatTriad(meshcat, "end_frame",
    #                 length=0.15, radius=0.006, X_PT=X_L7_end)

    # Add the iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(controller_plant,
                                  kp=[500] * (num_iiwa_positions),  # - 1) + [100],  # * num_iiwa_positions,
                                  ki=[10] * (num_iiwa_positions),  # - 1) + [10],
                                  kd=[50] * (num_iiwa_positions),  # - 1) + [10],
                                  has_reference_acceleration=False))
    iiwa_controller.set_name("iiwa_controller")
    builder.Connect(plant.get_state_output_port(iiwa),
                    iiwa_controller.get_input_port_estimated_state())

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values
    # if not connected).
    torque_passthrough = builder.AddSystem(PassThrough([0]
                                                       * num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(),
                        "iiwa_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(iiwa))

    # Add discrete derivative to command velocities.
    # Is there a way to do this continuously? Or would I have to command torques
    # Possible ideas: integrator,
    # desired_state_from_position = builder.AddSystem(
    #     StateInterpolatorWithDiscreteDerivative(
    #         num_iiwa_positions, time_step, suppress_initial_transient=True))
    # desired_state_from_position.set_name("desired_state_from_position")
    # builder.Connect(desired_state_from_position.get_output_port(),
    #                 iiwa_controller.get_input_port_desired_state())
    builder.Connect(iiwa_position.get_output_port(),
                    iiwa_controller.get_input_port_desired_state())

    # builder.Connect(iiwa_position.get_output_port(),
    #                 desired_state_from_position.get_input_port())

    # Export commanded torques.
    builder.ExportOutput(adder.get_output_port(), "iiwa_torque_commanded")
    builder.ExportOutput(adder.get_output_port(), "iiwa_torque_measured")

    builder.ExportOutput(plant.get_generalized_contact_forces_output_port(iiwa),
                         "iiwa_torque_external")

    # Wsg controller.
    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    wsg_controller.set_name("wsg_controller")
    builder.Connect(wsg_controller.get_generalized_force_output_port(),
                    plant.get_actuation_input_port(wsg))
    builder.Connect(plant.get_state_output_port(wsg),
                    wsg_controller.get_state_input_port())
    # builder.ExportInput(wsg_controller.get_desired_position_input_port(),
    #                     "wsg_position")
    builder.ExportInput(wsg_controller.get_force_limit_input_port(),
                        "wsg_force_limit")
    wsg_mbp_state_to_wsg_state = builder.AddSystem(
        MakeMultibodyStateToWsgStateSystem())
    builder.Connect(plant.get_state_output_port(wsg),
                    wsg_mbp_state_to_wsg_state.get_input_port())
    builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(),
                         "wsg_state_measured")
    builder.ExportOutput(wsg_controller.get_grip_force_output_port(),
                         "wsg_force_measured")

    finger_setpoints = PiecewisePolynomial.ZeroOrderHold(
        [0, 6, 10], np.array([[0.1], [-0.1], [-0.1]]).T)
    wsg_traj_source = SimpleTrajectorySource(finger_setpoints)
    wsg_traj_source.set_name("schunk_traj_source")
    builder.AddSystem(wsg_traj_source)


    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                         "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

    # Attach trajectories
    builder.Connect(q_source.get_output_port(),
                    iiwa_position.get_input_port())
    builder.Connect(wsg_traj_source.get_output_port(),
                    wsg_controller.get_desired_position_input_port())

    state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    torque_logger = LogVectorOutput(adder.get_output_port(), builder)
    diagram = builder.Build()
    diagram.set_name("PlaceBot")

    string = diagram.GetGraphvizString()
    src = Source(string)
    src.render('place_graph.gz', view=False)

    return diagram, plant, meshcat, state_logger, torque_logger, obj_idx  # plant.GetBodyByName('cube', obj)


def AddIiwa(plant, collision_model="no_collision"):
    sdf_path = "models/iiwa7.sdf"
    parser = Parser(plant)
    iiwa = parser.AddModelFromFile(sdf_path)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    q0 = [0.0, 0.1, 0, -1.2, 0, 1.6, 0]
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


def AddTwoLinkIiwa(plant, q0=[0.1, -1.2]):
    urdf = FindResource("models/two_link_iiwa14.urdf")

    parser = Parser(plant)
    iiwa = parser.AddModelFromFile(urdf)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


def AddWsg(plant, iiwa_model_instance, roll=np.pi / 2.0, welded=False):
    parser = Parser(plant)
    if welded:
        gripper = parser.AddModelFromFile(
            FindResource("models/schunk_wsg_50_welded_fingers.sdf"), "gripper")
    else:
        gripper = parser.AddModelFromFile(
            FindResourceOrThrow(
                "drake/manipulation/models/"
                "wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"))

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, roll), [0, 0, 0.114])
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa_model_instance),
                     plant.GetFrameByName("body", gripper), X_7G)

    # Set initial positions
    q0 = [-0.015, 0.015]
    idx = 0
    joint_indices = plant.GetJointIndices(gripper)
    for joint_index in joint_indices:
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, PrismaticJoint):
            joint.set_default_translation(q0[idx])
            idx += 1
    return gripper


####################################################################################
# Modified from the AddWSG example in pydrake:
# https://github.com/RussTedrake/manipulation/blob/master/manipulation/scenarios.py
####################################################################################
def AddObject(plant: MultibodyPlant, iiwa_model_instance, object_name: str, roll: float = np.pi) -> object:
    """
    Add an object welded to the 7th link of the iiwa
    :type plant: object
    :param plant:
    :param iiwa_model_instance:
    :param object_name:
    :param roll:
    :return:
    """
    parser = Parser(plant)
    try:
        object = parser.AddModelFromFile(
            'models/006_mustard_bottle.sdf', object_name)
            # FindResourceOrThrow(
            #     'drake/manipulation/models/'
            #     f'ycb/sdf/{object_library[object_name]}'), object_name)
    except KeyError:
        raise KeyError(f'Cannot find {object_name} in the object library.')

    print(type(plant.get_joint(plant.GetJointIndices(iiwa_model_instance)[-1])))
    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, roll), [0, 0, 0.2])
    # TODO: transform to be between the gripper fingers

    joint_offset = FixedOffsetFrame(
        'offset',
        plant.GetFrameByName('iiwa_link_7'),  # usually 7
        X_7G,
    )
    # Might need to add the frame to the plant first
    plant.AddFrame(joint_offset)
    joint = RevoluteJoint(
        'object_joint',
        joint_offset,
        plant.GetFrameByName('base_link_mustard'),
        np.array([0, 0, 1]),
    )
    plant.AddJoint(joint)
    print('added joint')
    print(plant.num_joints())
    return object


def AddGraspedObject(plant: MultibodyPlant, iiwa_model_instance, object_name: str, roll: float = np.pi) -> object:
    """
    Add an object welded to the 7th link of the iiwa
    :type plant: object
    :param plant:
    :param iiwa_model_instance:
    :param object_name:
    :param roll:
    :return:
    """
    parser = Parser(plant)
    try:
        object = parser.AddModelFromFile(
            'models/cube.sdf', 'cube')  # object_name)
    except KeyError:
        raise KeyError(f'Cannot find {object_name} in the object library.')

    # plant.WeldFrames(plant.GetFrameByName("iiwa_link_0", iiwa_model_instance),
    #                  plant.GetFrameByName("cube", object_name))
    # TODO: transform to be between the gripper fingers

    # X_start_table = RigidTransform(RotationMatrix(), [0.5, 0, 0.0])
    return plant.GetBodyByName('cube', object)

def AddTable(plant: MultibodyPlant, iiwa_model_instance):
    """
    Add an object welded to the 7th link of the iiwa
    :type plant: object
    :param plant:
    """
    parser = Parser(plant)
    start_table = parser.AddModelFromFile(
        'models/table.sdf', 'start_table')
    end_table = parser.AddModelFromFile(
        'models/table.sdf', 'end_table')

    X_start_table = RigidTransform(RotationMatrix(), [0.5, 0, 0.0])
    X_end_table = RigidTransform(RotationMatrix(), [-0.5, 0, 0.0])

    plant.WeldFrames(plant.GetFrameByName("iiwa_link_0", iiwa_model_instance),
                     plant.GetFrameByName("table_base", start_table), X_start_table)

    plant.WeldFrames(plant.GetFrameByName("iiwa_link_0", iiwa_model_instance),
                     plant.GetFrameByName("table_base", end_table), X_end_table)

    return object


def AddRgbdSensor(builder,
                  scene_graph,
                  X_PC,
                  depth_camera=None,
                  renderer=None,
                  parent_frame_id=None):
    """
    Adds a RgbdSensor to to the scene_graph at (fixed) pose X_PC relative to
    the parent_frame.  If depth_camera is None, then a default camera info will
    be used.  If renderer is None, then we will assume the name 'my_renderer',
    and create a VTK renderer if a renderer of that name doesn't exist.  If
    parent_frame is None, then the world frame is used.
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not parent_frame_id:
        parent_frame_id = scene_graph.world_frame_id()

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer,
                                MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer, CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
                ClippingRange(near=0.1, far=10.0), RigidTransform()),
            DepthRange(0.1, 10.0))

    rgbd = builder.AddSystem(
        RgbdSensor(parent_id=parent_frame_id,
                   X_PB=X_PC,
                   depth_camera=depth_camera,
                   show_window=False))

    builder.Connect(scene_graph.get_query_output_port(),
                    rgbd.query_object_input_port())

    return rgbd
