import pydrake.symbolic as sym

def remove_terms_with_small_coefficients(expr, tol=1e-9):
    print(expr)
    # Base case
    if isinstance(expr, float):
        if expr < tol:
            return 0
        else:
            return expr
    elif isinstance(expr, sym.Variable):
        return expr

    print(expr.Unapply())
    fn, terms = expr.Unapply()
    print(fn)
    # Base case
    if fn.__name__ == '_reduce_mul':
        if isinstance(terms[0], float):
            if terms[0] < tol:
                return 0

    # Recursive case
    return fn([remove_terms_with_small_coefficients(expr, tol)])


def test_remove_small_terms():
    x = sym.Variable('x')
    test1 = remove_terms_with_small_coefficients(1e-10)
    print(test1)
    assert test1 == 0

    test2 = remove_terms_with_small_coefficients(sym.Expression(x))
    print(test2)
    assert test2 == x


if __name__ == '__main__':
    test_remove_small_terms()