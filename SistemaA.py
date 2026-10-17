import numpy as np

from newtons import newton, newton_discreto, newton_modificado


def F(x, y):
    F = np.zeros(2)
    F[0] = x**3 - 3 * x * y**2 - 1  # f1(x, y)
    F[1] = 3 * x**2 * y - y**3      # f2(x, y)
    return F


def J(x, y):
    J = np.zeros((2, 2))
    J[0][0] = 3 * x**2 - 3 * y**2   # df1/dx
    J[0][1] = -6 * x * y            # df1/dy
    J[1][0] = 6 * x * y             # df2/dx
    J[1][1] = 3 * x**2 - 3 * y**2   # df2/dy
    return J


def J_aprox(x, y):
    J = np.zeros((2, 2))
    h = 1e-2
    J[0][0] = (F(x + h, y)[0] - F(x, y)[0]) / h
    J[0][1] = (F(x, y + h)[0] - F(x, y)[0]) / h
    J[1][0] = (F(x + h, y)[1] - F(x, y)[1]) / h
    J[1][1] = (F(x, y + h)[1] - F(x, y)[1]) / h
    return J


RAIZES_A = np.array(
    [
        [1.0, 0.0],
        [-0.5, np.sqrt(3) / 2],
        [-0.5, -np.sqrt(3) / 2],
    ],
    dtype=float,
)


def _indice_raiz(ponto, raizes, tol):
    distancias = np.linalg.norm(raizes - ponto, axis=1)
    indice = np.argmin(distancias)
    if distancias[indice] < tol:
        return int(indice + 1)
    return 0


def _calcular_mapas(solver, jac, F, raizes, x_min, x_max, y_min, y_max, N=201, itmax=20, tol=1e-6):
    x_vals = np.linspace(x_min, x_max, N + 1)
    y_vals = np.linspace(y_min, y_max, N + 1)

    C_map = np.zeros((len(y_vals), len(x_vals)), dtype=int)
    M_map = np.zeros((len(y_vals), len(x_vals)), dtype=int)

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            try:
                x_sol, y_sol, it, _ = solver(x, y, jac, F, itmax=itmax, tol=tol)
                raiz = _indice_raiz(np.array([x_sol, y_sol], dtype=float), raizes, tol)
                if raiz:
                    C_map[i, j] = raiz
                    M_map[i, j] = it
            except np.linalg.LinAlgError:
                pass

    return C_map, M_map, x_vals, y_vals


def sistemaA(N=500, itmax=20, tol=1e-6):
    """Calcula os mapas dos três métodos de Newton para o Sistema A."""
    resultados = {
        "metodos": {},
        "raizes": RAIZES_A,
        "x_min": -1.5,
        "x_max": 1.5,
        "y_min": -1.5,
        "y_max": 1.5,
        "N": N,
        "itmax": itmax,
        "tol": tol,
    }

    metodos = {
        "Newton": (newton, J),
        "Newton Modificado": (newton_modificado, J),
        "Newton Discreto": (newton_discreto, J_aprox),
    }

    for nome, (solver, jac) in metodos.items():
        C_map, M_map, _, _ = _calcular_mapas(
            solver,
            jac,
            F,
            RAIZES_A,
            resultados["x_min"],
            resultados["x_max"],
            resultados["y_min"],
            resultados["y_max"],
            N=N,
            itmax=itmax,
            tol=tol,
        )
        resultados["metodos"][nome] = {
            "convergencia": C_map,
            "iteracoes": M_map,
        }

    return resultados


def main():
    print("Sistema A")
    sistemaA()
    print("acabou")


if __name__ == "__main__":
    print("Resolvendo sistemas não lineares")
    main()
