import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw

def calculate_f_ext(plant, context):

    q = plant.GetPositionsAndVelocities(context)

    tau_g = plant.CalcGravityGeneralizedForces(context)

    # There's 19 of them!
    F_J = plant.get_reaction_forces_output_port().Eval(context)[-1]

    # Is this correct?
    X_J_O = RigidTransform(RollPitchYaw([np.pi/2, 0, 0]), [0, 0.1, 0])

    F_O = X_J_O @ F_J

    print(f"reaction_force: {F_O.translational()} {F_O.rotational()}")