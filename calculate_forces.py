import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.all import RevoluteJoint
import pydrake.symbolic as sym

from manipulation.meshcat_cpp_utils import AddMeshcatTriad


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

    AddMeshcatTriad(meshcat, "joint_pose",
                    length=0.15, radius=0.006,
                    X_PT=plant.GetJointByName('body_welds_to_base_link_mustard').child_body().EvalPoseInWorld(context))

    p_B_Bcm = plant.GetBodyByName('base_link_mustard').CalcCenterOfMassInBodyFrame(context)
    X_W_O = plant.GetBodyByName('base_link_mustard').EvalPoseInWorld(context)

    # AddMeshcatTriad(meshcat, "joint_pose",
    #                 length=0.15, radius=0.006,
    #                 X_PT=X_O)
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

    return F_W

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

