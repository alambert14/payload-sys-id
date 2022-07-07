import numpy as np
import matplotlib.pyplot as plt
import pydrake.symbolic as sym
from pydrake.all import (
    Parser, AddMultibodyPlantSceneGraph, SpatialInertia_, RotationalInertia_, DiagramBuilder,
    FindResourceOrThrow,
)
from pydrake.multibody.tree import UnitInertia_
from tqdm import tqdm

from manipulation.scenarios import AddIiwa

def wrapper(func, args):
    return func(*args)

def remove_terms_with_small_coefficients(expr, tol=1e-9):
    # Base case
    if isinstance(expr, float):
        if expr < tol:
            return 0.
        else:
            return expr

    fn, terms = expr.Unapply()
    # Base case
    if len(terms) == 1:
        return terms[0]
    # Base case
    if fn.__name__ == '_reduce_mul':
        if isinstance(terms[0], float):
            if terms[0] < tol:
                return 0.
    elif fn.__name__ == 'truediv': # TODO: this doesn't seem to work
        if isinstance(terms[0], float):
            if terms[0] < tol:
                return 0.

    # if fn.__name__ != '_reduce_mul' and fn.__name__ != '_reduce_add' and fn.__name__ != 'truediv':
    #     print(fn)

    # Recursive case
    new_terms = [remove_terms_with_small_coefficients(term, tol) for term in terms]
    return wrapper(fn, new_terms)


def test_remove_small_terms():
    x = sym.Variable('x')
    test1 = remove_terms_with_small_coefficients(1e-10)
    # print(test1)
    assert test1 == 0

    test2 = remove_terms_with_small_coefficients(5 * x + 5e-10 * pow(x, 2))
    print(test2)
    assert test2 == 5 * x

'''
def calc_mass(plant, state_log, torque_log):
    t = state_log.sample_times()
    q = state_log.data()[:8, :]
    v = state_log.data()[8:, :]
    tau = torque_log.data()

    M = t.shape[0] - 1
    MM = 14 * M
    N = 1
    Wdata = np.zeros((MM, N))
    w0data = np.zeros((MM, 1))
    offset = 0
    for i in tqdm(range(M)):
        h = t[i + 1] - t[i]
        vd = (v[:, i + 1] - v[:, i]) / h

        W, alpha, w0 = calc_lumped_parameters(plant, q[:, i], v[:, i], vd, tau[:, i])

        # print(len(alpha))

        W = sym.Evaluate(W, {})
        w0 = sym.Evaluate(w0, {})

        # if W.shape[1] < Wdata.shape[1]:
        #     W = np.hstack((W, np.zeros((14, Wdata.shape[1] - W.shape[1]))))
        # print(W.shape)
        Wdata[offset:offset + 14, :] = W[:, 0].reshape((14, N))  # sym.Evaluate(W, {})
        w0data[offset:offset + 14] = w0
        # Wdata[offset:offset + 2, :] = W[:, 0].reshape((2, N))  # sym.Evaluate(W, {})
        # w0data[offset:offset + 2] = w0[0]  # sym.Evaluate(w0, {})
        offset += 14

    alpha_fit = np.linalg.lstsq(Wdata, -w0data, rcond=None)[0]

    return alpha_fit
'''

def calc_data_matrix(plant, state_log, torque_log, mass = None):
    print(state_log.data().shape)
    print(torque_log.data().shape)
    t = state_log.sample_times()
    q = state_log.data()[:8, :]
    v = state_log.data()[8:, :]
    tau = torque_log.data()

    M = t.shape[0] - 1
    MM = 14 * M
    N = 10
    Wdata = np.zeros((MM, N))
    w0data = np.zeros((MM, 1))
    offset = 0
    valid_iterations = 0
    alpha_all_iterations = np.zeros((M, N))
    for i in tqdm(range(M)):
        h = t[i+1] - t[i]
        vd = (v[:, i + 1] - v[:, i]) / h

        W, alpha, w0, params = calc_lumped_parameters(plant, q[:, i], v[:, i], vd, tau[:, i], mass=mass)

        m, cx, cy, cz, G_0, G_1, G_2, G_3, G_4, G_5 = params
        expected_alpha = [
            sym.Expression(m),
            (G_0 * m), (G_1 * m), (G_2 * m),
            (G_3 * m), (G_4 * m), (G_5 * m),
            (m * cx), (m * cy), (m * cz),
        ]

        try:
            assert all([alpha[i].EqualTo(expected_alpha[i]) for i in range(len(expected_alpha))])
        except AssertionError:
            print('Inconsistent lumped parameters: ', alpha)
            continue
        # print(alpha)

        W = sym.Evaluate(W, {})
        w0 = sym.Evaluate(w0, {})

        # if W.shape[1] < Wdata.shape[1]:
        #     W = np.hstack((W, np.zeros((14, Wdata.shape[1] - W.shape[1]))))

        try:
            Wdata[offset:offset+14, :] = W  # sym.Evaluate(W, {})
            w0data[offset:offset+14] = w0  # sym.Evaluate(w0, {})
            offset += 14
            valid_iterations += 1
        except ValueError:
            pass
        alpha_fit = np.linalg.lstsq(Wdata[:valid_iterations], -w0data[:valid_iterations], rcond=None)[0]
        alpha_all_iterations[i, :] = alpha_fit.squeeze()

    return alpha_all_iterations

def plot_parameter_est(data, index, parameter: str, ground_truth, color = 'blue'):
    plt.xlabel('Timestep in trajectory ($t_0 = $200)')
    plt.ylabel(f'Least-squares estimation of {parameter}')


    result = data[200:, index]
    if index in range(7, 10):
        result /= data[-1, 0]
    mse_error = abs(ground_truth - result[-1])
    plt.title(f'Estimation of {parameter} during trajectory \n'
              f'True value $=$ {round(ground_truth, 6)}, Estimated $=$ {round(result[-1], 6)}, Error $=$ {round(mse_error, 6)}')
    plt.plot(result, color=color)
    plt.plot([ground_truth] * (data.shape[0] - 200), '--', color=color)
    # plt.yscale('log')
    plt.show()

def plot_all_parameters_est(data, ground_truth):
    plot_parameter_est(data, 0, 'mass $m$', ground_truth[0], color='black')

    plot_parameter_est(data, 7, 'center of mass $c_{m_x}$', ground_truth[1], color='red')
    plot_parameter_est(data, 8, 'center of mass $c_{m_y}$', ground_truth[2], color='green')
    plot_parameter_est(data, 9, 'center of mass $c_{m_z}$', ground_truth[3], color='blue')

    plot_parameter_est(data, 1, 'moment of inertia $I_{xx}$', ground_truth[4], color='red')
    plot_parameter_est(data, 2, 'moment of inertia $I_{yy}$', ground_truth[5], color='green')
    plot_parameter_est(data, 3, 'moment of inertia $I_{zz}$', ground_truth[6], color='blue')
    plot_parameter_est(data, 4, 'product of inertia $I_{xy}$', ground_truth[7], color='orange')
    plot_parameter_est(data, 5, 'product of inertia $I_{xz}$', ground_truth[8], color='purple')
    plot_parameter_est(data, 6, 'product of inertia $I_{yz}$', ground_truth[9], color='teal')


def calc_lumped_parameters(plant, q, v, vd, tau, mass = None):
    context = plant.CreateDefaultContext()
    sym_plant = plant.ToSymbolic()
    sym_context = sym_plant.CreateDefaultContext()
    sym_context.SetTimeStateAndParametersFrom(context)
    sym_plant.FixInputPortsFrom(plant, context, sym_context)

    state = sym_context.get_continuous_state()

    # State variables
    # q = sym.MakeVectorVariable(state.num_q(), "q")
    # v = sym.MakeVectorVariable(state.num_v(), "v")
    # qd = sym.MakeVectorVariable(state.num_q(), "\dot{q}")
    # vd = sym.MakeVectorVariable(state.num_v(), "\dot{v}")
    # tau = sym.MakeVectorVariable(state.num_q() - 1, 'u')
    # q = np.ones(state.num_q()) * np.pi / 4
    # v = np.ones(state.num_v()) * np.pi / 4
    # qd = np.ones(state.num_q()) * np.pi / 4
    # vd = np.ones(state.num_v()) * np.pi / 4
    # tau = np.ones(state.num_q() - 1) * np.pi / 4
    full_q = []
    full_v = []
    for state in q:
        full_q.append(state)
    full_q.append(np.pi / 4)
    for vel in v:
        full_v.append(vel)
    full_v.append(np.pi / 4)
    full_q = np.array(full_q)
    full_v = np.array(full_v)
    # print('num q: ', state.num_q())
    # print('num v: ', state.num_v())

    # Parameters
    G = sym.MakeVectorVariable(6, 'G')  # Inertia tensor/mass matrix
    m = sym.Variable('m')  # mass
    cx = sym.Variable('cx')  # center of mass
    cy = sym.Variable('cy')
    cz = sym.Variable('cz')

    # if mass is None:
    # m = sym.Variable('m')
    # else:
    #     m = mass

    sym_plant.get_actuation_input_port().FixValue(sym_context, tau)
    sym_plant.SetPositions(sym_context, q)
    sym_plant.SetVelocities(sym_context, v)

    obj = sym_plant.GetBodyByName('base_link_mustard')
    #                               mass, origin to Com, RotationalInertia
    inertia = SpatialInertia_[sym.Expression](m, [cx, cy, cz],
                                              UnitInertia_[sym.Expression](
                                                 G[0], G[1], G[2], G[3], G[4], G[5]))

    #  Test RotationalInertia
    # test_inertia = RotationalInertia_[sym.Expression](
    #     m, [0.1, 0.01, 0.03]
    # )
    # test_inertia = UnitInertia_[sym.Expression](
    #     m, [0.1, 0.01, 0.03]
    # )
    # print(test_inertia.get_moments(), test_inertia.get_products())

    obj.SetSpatialInertiaInBodyFrame(sym_context, inertia)

    derivatives = sym_context.Clone().get_mutable_continuous_state()
    derivatives.SetFromVector(np.hstack((0 * v, vd)))
    # print(type(sym_plant), type(derivatives), type(sym_context))
    residual = sym_plant.CalcImplicitTimeDerivativesResidual(
        sym_context, derivatives)
    # print('symbolic equation: ', residual)
    # eq = Math(ToLatex(residual[2:], 2))
    # with open("equation.png", "wb+") as png:
    #    print(type(eq.image))
    #    png.write(eq.image)

    # print('getting lumped parameters...')
    params = [cx, cy, cz, G[0], G[1], G[2], G[3], G[4], G[5]]
    if mass is None:
        params = [m] + params
    W, alpha, w0 = sym.DecomposeLumpedParameters(residual[2:], [m, cx, cy, cz, G[0], G[1], G[2], G[3], G[4], G[5]])
    # simp_alpha = [remove_terms_with_small_coefficients(expr, 1e-3) for expr in alpha]

    return W, alpha, w0, params


def calc_lumped_parameters_stack_overflow():
    # Create the plant
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=0)
    Parser(plant, scene_graph).AddModelFromFile(
        FindResourceOrThrow("drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    plant.Finalize()
    diagram = builder.Build()

    context = plant.CreateDefaultContext()
    sym_plant = plant.ToSymbolic()
    sym_context = sym_plant.CreateDefaultContext()
    sym_context.SetTimeStateAndParametersFrom(context)
    sym_plant.FixInputPortsFrom(plant, context, sym_context)

    state = sym_context.get_continuous_state()

    # Random state/command inputs
    # (Currently these are recorded from the robot executing a trajectory)
    q = np.random.random(size=state.num_q())
    v = np.random.random(size=state.num_v())
    vd = np.random.random(size=state.num_v())
    tau = np.random.random(size=state.num_q())  # Remove -1  for fully actuated system

    # Parameters
    I = sym.MakeVectorVariable(6, 'I')  # Inertia tensor/mass matrix
    m = sym.Variable('m')  # mass
    cx = sym.Variable('cx')  # center of mass
    cy = sym.Variable('cy')
    cz = sym.Variable('cz')

    sym_plant.get_actuation_input_port().FixValue(sym_context, tau)
    sym_plant.SetPositions(sym_context, q)
    sym_plant.SetVelocities(sym_context, v)

    obj = sym_plant.GetBodyByName('iiwa_link_7')
    inertia = SpatialInertia_[sym.Expression](m, [cx, cy, cz],
                                              UnitInertia_[sym.Expression](
                                                  I[0], I[1], I[2], I[3], I[4], I[5]))
    obj.SetSpatialInertiaInBodyFrame(sym_context, inertia)

    derivatives = sym_context.Clone().get_mutable_continuous_state()
    derivatives.SetFromVector(np.hstack((0 * v, vd)))
    residual = sym_plant.CalcImplicitTimeDerivativesResidual(
        sym_context, derivatives)

    W, alpha, w0 = sym.DecomposeLumpedParameters(residual[2:],
                                                 [m, cx, cy, cz, I[0], I[1], I[2], I[3], I[4], I[5]])

    return W, alpha, w0


def detect_slip(plant, state_log):
    times = state_log.sample_times()
    T = len(times)
    data = state_log.data()

    last_pose_diff = 0
    last_velocity = 0
    pose_diffs = []
    vel_diffs = []
    for i, t in enumerate(times):
        # Pass the state data into a default context to pass into EvalBodyPoseInWorld
        temp_context = plant.CreateDefaultContext()

        # print(plant.num_positions())
        plant.SetPositions(temp_context, data[:16, i])
        temp_context.SetTime(t)

        eef = plant.GetBodyByName('iiwa_link_7')
        eef_pose = plant.EvalBodyPoseInWorld(temp_context, eef)

        obj = plant.GetBodyByName('cube')
        obj_pose = plant.GetFreeBodyPose(temp_context, obj)  # Is it no longer a free body once grasped by the robot?

        z_diff = eef_pose.translation()[2] - obj_pose.translation()[2]
        velocity = (z_diff - last_pose_diff) / (t - times[i - 1]) if i > 0 else 0
        if abs(velocity) > 4:
            velocity = 0

        print(velocity)
        # print(z_diff)
        if t > 0:
            if abs(last_velocity - velocity) / (t - times[i - 1]) > 5:
                print(f'slippage of some kind at time {t}')

        pose_diffs.append(z_diff)
        vel_diffs.append(velocity)
        last_pose_diff = z_diff
        last_velocity = velocity

    plt.plot(times, pose_diffs)
    plt.plot(times, vel_diffs)
    plt.show()

if __name__ == '__main__':
    # test_remove_small_terms()
    print(calc_lumped_parameters_stack_overflow())