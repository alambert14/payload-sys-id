import pydrake.symbolic as sym

def wrapper(func, args):
    return func(*args)

def remove_terms_with_small_coefficients(expr, tol=1e-9):
    # print(type(expr))
    # print('recurse')
    # Base case
    if isinstance(expr, float):
        if expr < tol:
            return 0
        else:
            return expr
    elif isinstance(expr, sym.Variable):
        print('Variable')
        return expr

    # print(expr.Unapply())
    fn, terms = expr.Unapply()
    # print(fn)
    # print('terms: ', terms)
    # Base case
    if len(terms) == 1:
        print('terms == 1')
        return terms[0]
    # Base case
    if fn.__name__ == '_reduce_mul':
        if isinstance(terms[0], float):
            if terms[0] < tol:
                return 0

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


if __name__ == '__main__':
    test_remove_small_terms()