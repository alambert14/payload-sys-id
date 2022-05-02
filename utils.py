import pydrake.symbolic as sym
from pandas import DataFrame


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

def calc_data_matrix(log, plant):
    data = DataFrame({
        't': log.sample_times(),
        'u': utraj.vector_values(log.sample_times())[0, :],
        'q': log.data()[0:-2:2, :],
        'theta': log.data()[-2, :],
        'qdot': log.data()[1:-2:2, :],
        'thetadot': log.data()[-1, :],
    })

    M = data.t.shape[0] - 1
    MM = 2 * M * len(data)
    N = 8
    Wdata = np.zeros((MM, N))
    w0data = np.zeros((MM, 1))
    offset = 0
    for d in data:
        for i in range(M):
            h = d.t[i + 1] - d.t[i]
            env = {
                q[0]: d.x[i],
                q[1]: d.theta[i],
                v[0]: d.xdot[i],
                v[1]: d.thetadot[i],
                vd[0]: (d.xdot[i + 1] - d.xdot[i]) / h,
                vd[1]: (d.thetadot[i + 1] - d.thetadot[i]) / h,
                tau[0]: d.u[i],
            }

            Wdata[offset:offset + 2, :] = Evaluate(W, env)
            w0data[offset:offset + 2] = Evaluate(w0, env)
            offset += 2

    print(Wdata.shape)
    alpha_fit = np.linalg.lstsq(Wdata, -w0data, rcond=None)[0]
    alpha_true = Evaluate(alpha, {mc: 10, mp: 1, l: .5})


if __name__ == '__main__':
    test_remove_small_terms()