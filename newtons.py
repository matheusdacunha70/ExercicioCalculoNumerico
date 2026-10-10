import numpy as np


def newton(x0, y0, Jac, F, itmax=20, tol=1e-6):
    x = x0
    y = y0
    erro = float('inf')
    it = 0

    while erro > tol and it < itmax:
        jac = Jac(x, y)
        if np.linalg.norm(jac) == 0:
            raise np.linalg.LinAlgError("Jacobiana singular no ponto inicial")

        s = np.linalg.solve(jac, -F(x, y))
        x = x + s[0]
        y = y + s[1]
        erro = np.linalg.norm(s)
        it += 1

    return x, y, it, erro


def newton_modificado(x0, y0, Jac, F, itmax=20, tol=1e-6):
    x = x0
    y = y0
    erro = float('inf')
    it = 0

    jac = Jac(x, y)
    while erro > tol and it < itmax:
        s = np.linalg.solve(jac, -F(x, y))
        x = x + s[0]
        y = y + s[1]
        erro = np.linalg.norm(s)
        it += 1

    return x, y, it, erro


def newton_discreto(x0, y0, Jac, F, itmax=20, tol=1e-6):
    pass
