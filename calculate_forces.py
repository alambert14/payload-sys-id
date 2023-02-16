import numpy as np
from manipulation.utils import AddPackagePaths
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.all import RevoluteJoint, DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, JacobianWrtVariable

import pydrake.symbolic as sym

from manipulation.meshcat_cpp_utils import AddMeshcatTriad
from pydrake.multibody.parsing import ProcessModelDirectives, LoadModelDirectives

from make_iiwa_and_object import add_package_paths_local


def calculate_f_ext(plant, context, meshcat):

    q = plant.GetPositionsAndVelocities(context)

    tau_g = plant.CalcGravityGeneralizedForces(context)
    print(f'gravity {plant.gravity_field().gravity_vector()}')

    #print(f"Force elements: {plant.num_force_elements()}")
    # print(f"GetForceElement: {plant.GetForceElement(0)}")

    # Set q_dots to zero
    # iiwa = plant.GetModelInstanceByName('iiwa')
    # for joint_index in plant.GetJointIndices(iiwa):
    #     joint = plant.get_mutable_joint(joint_index)
    #     if isinstance(joint, RevoluteJoint):
    #         joint.set_angular_rate(context, 0.)

    # There's 12 of them!
    F_J = plant.get_reaction_forces_output_port().Eval(context)[-1]

    AddMeshcatTriad(meshcat, "joint_pose",
                    length=0.15, radius=0.006,
                    X_PT=plant.GetJointByName('body_welds_to_base_link_mustard').child_body().EvalPoseInWorld(context))
    # Is this correct?
    # X_J_O = RigidTransform(RollPitchYaw([np.pi/2, 0, 0]), [0, 0.1, 0])
    # X_O_J = X_J_O.inverse()

    p_B_Bcm = plant.GetBodyByName('base_link_mustard').CalcCenterOfMassInBodyFrame(context)

    X_O = plant.GetBodyByName('base_link_mustard').EvalPoseInWorld(context)
    # X_O.set_translation(X_O.translation() + p_B_Bcm)
    # AddMeshcatTriad(meshcat, "joint_pose",
    #                 length=0.15, radius=0.006,
    #                 X_PT=X_O)
    # Seems like the center of mass is very wrong

    # F_O is closest to pure gravitational force
    F_O = F_J.Shift(p_B_Bcm)

    # X_O_adj = np.zeros((6, 6))
    # X_O_adj[:3, :3] = X_O.inverse().rotation().matrix()
    # X_O_adj[3:, :3] = X_O.inverse().translation().dot(X_O.inverse().rotation().matrix())
    # X_O_adj[3:, 3:] = X_O.inverse().rotation().matrix()
    #
    # F_O_np = np.zeros(6)
    # F_O_np[:3] = F_O.rotational()
    # F_O_np[3:] = F_O.translational()

    # This is correct
    F_W = X_O.rotation().matrix().dot(F_O.translational())

    print(np.linalg.norm(F_W))

    # When center of mass is zero, F_J == F_O
    print(f"reaction_force: {F_W}")  #{F_J.translational()} {F_J.rotational()}")

def calculate_gravity_from_wrench(plant, context, meshcat):
    """
    Given a plant and context, calculate the wrench at the center of mass of an object held in the gripper
    :param plant:
    :param context:
    :param meshcat:
    :return:
    """
    F_J = plant.get_reaction_forces_output_port().Eval(context)[-1]

    AddMeshcatTriad(meshcat, "joint_child_pose",
                    length=0.15, radius=0.006,
                    X_PT=plant.GetJointByName('body_welds_to_base_link_mustard').child_body().EvalPoseInWorld(context))

    p_B_Bcm = plant.GetBodyByName('base_link_mustard').CalcCenterOfMassInBodyFrame(context)
    X_W_O = plant.GetBodyByName('base_link_mustard').EvalPoseInWorld(context)

    F_O = F_J.Shift(p_B_Bcm)

    X_O_rot = np.zeros((6, 6))
    X_O_rot[:3, :3] = X_W_O.rotation().matrix()
    X_O_rot[3:, 3:] = X_W_O.rotation().matrix()

    F_O_np = np.zeros(6)
    F_O_np[:3] = F_O.rotational()
    F_O_np[3:] = F_O.translational()

    # This is correct
    F_W = X_O_rot.dot(F_O_np)

    obj_mass = plant.GetBodyByName('base_link_mustard').get_mass(context)
    exp_gravity = -obj_mass * plant.gravity_field().gravity_vector()[2]
    # When center of mass is zero, F_J == F_O
    print(f"reaction_force: {F_W}")  # {F_J.translational()} {F_J.rotational()}")
    print(f"Check that force norm = {exp_gravity}: {np.linalg.norm(F_W[3:])}")


    # try to cancel out torques from arm:
    F_g_E = calculate_gravity_of_iiwa(q0=[0., 0., 0., -np.pi/2, 0., 0., 0.])
    X_W_E = plant.GetBodyByName('body').EvalPoseInWorld(context)
    # X_E_O = X_W_E.inverse() @ X_W_O

    # F_g_O = F_g_E.Shift(X_E_O.translation())  # not just yet, only shift

    X_W_E_rot = np.zeros((6, 6))
    X_W_E_rot[:3, :3] = X_W_E.rotation().matrix()
    X_W_E_rot[3:, 3:] = X_W_E.rotation().matrix()

    # F_g_E_np = np.zeros(6)
    # F_g_E_np[:3] = F_g_E.rotational()
    # F_g_E_np[3:] = F_g_E.translational()

    F_g_W = X_W_E_rot.dot(F_g_E)

    F_W_total = F_g_W + F_W

    print(f'total reaction_force: {F_W_total}')
    print(f"Check that force norm = {exp_gravity}: {np.linalg.norm(F_W_total[3:])}")


    # Try getting total forces of the system, then use Jacobian to find all forces at the end-effector
    # Is the total forces just gravity at the moment? Since there is no motion we aren't in equilibrium
    tau_g = plant.CalcGravityGeneralizedForces(context)
    X_W = plant.world_frame()
    X_O = plant.GetBodyByName("base_link_mustard").body_frame()
    J_O = plant.CalcJacobianSpatialVelocity(context, JacobianWrtVariable.kQDot,
                                            X_O, [0, 0, 0], X_W, X_W)  # 6 x 7

    f_g_O = J_O.dot(tau_g)
    print(f"Gravity at object? {f_g_O}")
    # hoooooly shit its almost the same as the reaction forces yippee
    # That's a win for today :)

    return F_W

def calculate_gravity_of_iiwa(q0=[0., 0., 0., 0., 0., 0., 0.]):
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=2e-2)
    parser = Parser(plant)

    AddPackagePaths(parser)  # Russ's manipulation repo.
    add_package_paths_local(parser)  # local.

    ProcessModelDirectives(LoadModelDirectives('models/iiwa_and_schunk.yml'), plant, parser)
    iiwa = plant.GetModelInstanceByName('iiwa')

    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    context_plant = plant.GetMyContextFromRoot(context)

    X_W = plant.world_frame()
    X_E = plant.GetBodyByName("body").body_frame()
    J_E = plant.CalcJacobianSpatialVelocity(context_plant, JacobianWrtVariable.kQDot,
                                            X_E, [0, 0, 0], X_W, X_W)  # 6 x 7

    M = plant.CalcMassMatrix(context_plant)  # 7 x 7

    # Check that this is correct
    M_E = np.linalg.inv(J_E.dot(np.linalg.inv(M).dot(J_E.T)))  # 6 x 6

    tau_g = plant.CalcGravityGeneralizedForces(context_plant)  # 7 x 1

    f_g_B = M_E.dot(J_E.dot(np.linalg.inv(M))).dot(tau_g)  # 6 x 1
    print(f'f_g_E: {f_g_B}')
    # Results seem very not good when robot is pointed straight up

    return f_g_B


def identify_mass_and_cm(plant, context, meshcat):
    """

    :param plant:
    :param context:
    :param meshcat:
    :return:
    """
    sym_plant = plant.ToSymbolic()
    sym_context = sym_plant.CreateDefaultContext()
    sym_context.SetTimeStateAndParametersFrom(context)
    sym_plant.FixInputPortsFrom(plant, context, sym_context)

    # Parameters
    # I = sym.MakeVectorVariable(6, 'I')  # Inertia tensor/mass matrix
    m = sym.Variable('m')  # mass
    cx = sym.Variable('cx')  # center of mass
    cy = sym.Variable('cy')
    cz = sym.Variable('cz')

    obj = sym_plant.GetBodyByName('base_link_mustard')
    obj.SetMass(m)
    obj.SetCenterOfMassInBodyFrame([cx, cy, cz])

