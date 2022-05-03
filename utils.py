from pandas import DataFrame
import pydrake.symbolic as sym
import numpy as np
from pydrake.multibody.tree import SpatialInertia_, RotationalInertia_
from tqdm import tqdm

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

def calc_data_matrix(plant, state_log, torque_log):
    print(state_log.data().shape)
    print(torque_log.data().shape)
    t = state_log.sample_times()
    q = state_log.data()[:8, :]
    v = state_log.data()[8:, :]
    tau = torque_log.data()

    M = t.shape[0] - 1
    MM = 14 * M
    N = 19
    Wdata = np.zeros((MM, N))
    w0data = np.zeros((MM, 1))
    offset = 0
    for i in tqdm(range(M)):
        h = t[i+1] - t[i]
        vd = (v[:, i + 1] - v[:, i]) / h

        W, alpha, w0 = calc_lumped_parameters(plant, q[:, i], v[:, i], vd, tau[:, i])

        # print(len(alpha))

        W = sym.Evaluate(W, {})
        w0 = sym.Evaluate(w0, {})

        if W.shape[1] < Wdata.shape[1]:
            W = np.hstack((W, np.zeros((14, Wdata.shape[1] - W.shape[1]))))
        Wdata[offset:offset+14, :] = W # sym.Evaluate(W, {})
        w0data[offset:offset+14] = w0 # sym.Evaluate(w0, {})
        offset += 14

    alpha_fit = np.linalg.lstsq(Wdata, -w0data, rcond=None)[0]

    return alpha_fit


def calc_lumped_parameters(plant, q, v, vd, tau):
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
    # q = np.ones(state.num_q()) * np.pi / 4
    # v = np.ones(state.num_v()) * np.pi / 4
    # qd = np.ones(state.num_q()) * np.pi / 4
    # vd = np.ones(state.num_v()) * np.pi / 4
    # tau = np.ones(state.num_q() - 1) * np.pi / 4

    # print('num q: ', state.num_q())
    # print('num v: ', state.num_v())

    # Parameters
    I = sym.MakeVectorVariable(6, 'I')  # Inertia tensor/mass matrix
    m = sym.Variable('m')  # mass
    cx = sym.Variable('cx')  # center of mass
    cy = sym.Variable('cy')
    cz = sym.Variable('cz')

    sym_plant.get_actuation_input_port().FixValue(sym_context, tau)
    sym_plant.SetPositions(sym_context, q)
    sym_plant.SetVelocities(sym_context, v)

    obj = sym_plant.GetBodyByName('base_link_mustard')
    #                               mass, origin to Com, RotationalInertia
    inertia = SpatialInertia_[sym.Expression].MakeFromCentralInertia(m, [cx, cy, cz],
                                                                 RotationalInertia_[sym.Expression](
                                                                     I[0], I[1], I[2], I[3], I[4], I[5]))
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
    W, alpha, w0 = sym.DecomposeLumpedParameters(residual[2:],
                                                 [m, cx, cy, cz, I[0], I[1], I[2], I[3], I[4], I[5]])

    # print(remove_terms_with_small_coefficients(alpha[1]))
    simp_alpha = [remove_terms_with_small_coefficients(expr, 1e-3) for expr in alpha]

    return W, simp_alpha, w0


if __name__ == '__main__':
    test_remove_small_terms()