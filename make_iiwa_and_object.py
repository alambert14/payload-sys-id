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

from pydrake.all import (Adder, AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, Demultiplexer,
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
from pydrake.multibody.tree import RevoluteJoint, FixedOffsetFrame, SpatialInertia_, RotationalInertia, \
    RotationalInertia_
from pydrake.symbolic import Expression, MakeVectorVariable, MakeMatrixVariable, Variable, DecomposeLumpedParameters
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.geometry import Meshcat
from pydrake.systems.primitives import TrajectorySource, LogVectorOutput
from pydrake.trajectories import PiecewisePolynomial

from sysid_trajectory import PickAndPlaceTrajectorySource
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
    AddObject(plant, iiwa, object_name)
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
    AddWsg(controller_plant, controller_iiwa, welded=True)

    controller_plant.Finalize()

    # Create sample trajectory
    q_knots = np.array([# [0, 1.5, 0, 0],
                        #  [3., 1.5, 0, 0]])
                        [1.57, 0., 0., -1.57, 0., 1.57, 0,
                         0, 0, 0, 0, 0, 0, 0],
        [1.57, 0., 0., -1.57, 0., 1.57, 0,
         0, 0, 0, 0, 0, 0, 0]
    ])
    traj = PiecewisePolynomial.ZeroOrderHold([0, 1], q_knots.T)
    q_source = builder.AddSystem(TrajectorySource(traj))
    X_L7_start = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 0)), [0.6, 0, 0.6])
    X_L7_end = RigidTransform(RotationMatrix(RollPitchYaw(0, 3.14, 0)), [-0.6, 0, 0.4])
    # q_source = builder.AddSystem(PickAndPlaceTrajectorySource(plant, X_L7_start, X_L7_end))
    AddMeshcatTriad(meshcat, "start_frame",
                    length=0.15, radius=0.006, X_PT=X_L7_start)
    AddMeshcatTriad(meshcat, "end_frame",
                    length=0.15, radius=0.006, X_PT=X_L7_end)

    # Add the iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(controller_plant,
                                  kp=[100] * num_iiwa_positions,
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
    # wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    # wsg_controller.set_name("wsg_controller")
    # builder.Connect(wsg_controller.get_generalized_force_output_port(),
    #                 plant.get_actuation_input_port(wsg))
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
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    # viz = ConnectMeshcatVisualizer(
    #     builder, scene_graph, zmq_url='tcp://127.0.0.1:6000', prefix="environment")
    MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

    # Attach trajectory
    builder.Connect(q_source.get_output_port(),
                    iiwa_position.get_input_port())

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    diagram = builder.Build()
    diagram.set_name("ManipulationStation")

    string = diagram.GetGraphvizString()
    src = Source(string)
    src.render('graph.gz', view=False)


    print(calc_lumped_parameters(plant))

    return diagram, meshcat, logger


def calc_lumped_parameters(plant):
    context = plant.CreateDefaultContext()
    sym_plant = plant.ToSymbolic()
    sym_context = sym_plant.CreateDefaultContext()
    sym_context.SetTimeStateAndParametersFrom(context)
    sym_plant.FixInputPortsFrom(plant, context, sym_context)

    state = sym_context.get_continuous_state()

    # State variables
    # q = MakeVectorVariable(state.num_q(), "q")
    # v = MakeVectorVariable(state.num_v(), "v")
    # qd = MakeVectorVariable(state.num_q(), "\dot{q}")
    # vd = MakeVectorVariable(state.num_v(), "\dot{v}")
    # tau = MakeVectorVariable(1, 'u')
    q = np.ones(state.num_q()) * np.pi / 4
    v = np.ones(state.num_v()) * np.pi / 4
    qd = np.ones(state.num_q()) * np.pi / 4
    vd = np.ones(state.num_v()) * np.pi / 4
    tau = np.ones(state.num_q() - 1) * np.pi / 4

    # Parameters
    I = MakeVectorVariable(6, 'I')  # Inertia tensor/mass matrix
    m = Variable('m')  # mass
    cx = Variable('cx')  # center of mass
    cy = Variable('cy')
    cz = Variable('cz')

    sym_plant.get_actuation_input_port().FixValue(sym_context, tau)
    sym_plant.SetPositions(sym_context, q)
    sym_plant.SetVelocities(sym_context, v)

    obj = sym_plant.GetBodyByName('base_link_mustard')
    #                               mass, origin to Com, RotationalInertia
    inertia = SpatialInertia_[Expression].MakeFromCentralInertia(m, [cx, cy, cz],
        RotationalInertia_[Expression](
            I[0], I[1], I[2], I[3], I[4], I[5]))
    obj.SetSpatialInertiaInBodyFrame(sym_context, inertia)

    derivatives = sym_context.Clone().get_mutable_continuous_state()
    derivatives.SetFromVector(np.hstack((0*v, vd)))
    # print(type(sym_plant), type(derivatives), type(sym_context))
    residual = sym_plant.CalcImplicitTimeDerivativesResidual(
        sym_context, derivatives)
    # print('symbolic equation: ', residual)
    #eq = Math(ToLatex(residual[2:], 2))
    #with open("equation.png", "wb+") as png:
    #    print(type(eq.image))
    #    png.write(eq.image)

    print('getting lumped parameters...')
    W, alpha, w0 = DecomposeLumpedParameters(residual[2:],
         [m, cx, cy, cz, I[0], I[1], I[2], I[3], I[4], I[5]])

    # print(remove_terms_with_small_coefficients(alpha[1]))
    simp_alpha = [remove_terms_with_small_coefficients(expr, 1e-3) for expr in alpha]

    return W, simp_alpha, w0


def AddIiwa(plant, collision_model="no_collision"):
    sdf_path = "models/iiwa7.sdf" # "models/two_DOF_iiwa.sdf" # "models/iiwa7.sdf"

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
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_0", iiwa_model_instance),
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
