import numpy as np


def F(x, y):
    F = np.zeros(2)
    F[0] = x**5 - 10*x**3 * y**2 + 5*x*y**4 - 1
    F[1] = 5 * x**4 * y - 10*x**2 * y**3 + y**5
    return F

def J(x, y):
    J = np.zeros((2,2))
    J[0][0] = 5*x**4 - 30*x**2 * y**2 + 5*y**4
    J[0][1] = -20 * x**3 * y + 20*x*y**3
    J[1][0] = 20 * x**3*y - 20*x * y**3
    J[1][1] = 5 * x**4 - 30*x**2 * y**2 + 5*y**4
    return J

def J_aprox(x, y):
    J = np.zeros((2,2))
    h = 1e-2
    J[0][0] = (F(x+h,y)[0] - F(x,y)[0])/h
    J[0][1] = (F(x,y+h)[0] - F(x,y)[0])/h
    J[1][0] = (F(x+h,y)[1] - F(x,y)[1])/h
    J[1][1] = (F(x,y+h)[1] - F(x,y)[1])/h
    return J

RAIZES_B = np.array(
    [
        [1.0, 0.0],
        [np.cos(2 * np.pi / 5), np.sin(2 * np.pi / 5)],
        [np.cos(4 * np.pi / 5), np.sin(4 * np.pi / 5)],
        [np.cos(6 * np.pi / 5), np.sin(6 * np.pi / 5)],
        [np.cos(8 * np.pi / 5), np.sin(8 * np.pi / 5)],
    ],
    dtype=float,
)


def _indice_raiz(ponto, raizes, raiz_tol):
    distancias = np.linalg.norm(raizes - ponto, axis=1)
    indice = np.argmin(distancias)
    if distancias[indice] < raiz_tol:
        return int(indice + 1)
    return 0


def _newton_com_criterio_raiz(x0, y0, jac_func, func, raizes, itmax=20, tol=1e-6, raiz_tol=5e-2):
    x_k = np.array([x0, y0], dtype=float)
    iter_count = 0

    for _ in range(itmax):
        fx = func(x_k[0], x_k[1])

        if np.linalg.norm(fx) < tol:
            break

        jac = jac_func(x_k[0], x_k[1])

        try:
            delta = np.linalg.solve(jac, -fx)
        except np.linalg.LinAlgError:
            break

        x_k = x_k + delta
        iter_count += 1

        raiz = _indice_raiz(x_k, raizes, raiz_tol)
        if raiz:
            return raiz, iter_count, True

    return 0, iter_count, False


def _newton_modificado_com_criterio_raiz(x0, y0, jac_func, func, raizes, itmax=20, tol=1e-6, raiz_tol=5e-2):
    x_k = np.array([x0, y0], dtype=float)
    iter_count = 0
    jac = jac_func(x_k[0], x_k[1])

    for _ in range(itmax):
        fx = func(x_k[0], x_k[1])

        if np.linalg.norm(fx) < tol:
            break

        try:
            delta = np.linalg.solve(jac, -fx)
        except np.linalg.LinAlgError:
            break

        x_k = x_k + delta
        iter_count += 1

        raiz = _indice_raiz(x_k, raizes, raiz_tol)
        if raiz:
            return raiz, iter_count, True

    return 0, iter_count, False


def _metodo_newton(x0, y0, itmax=20, tol=1e-6, raiz_tol=5e-2):
    return _newton_com_criterio_raiz(x0, y0, J, F, RAIZES_B, itmax=itmax, tol=tol, raiz_tol=raiz_tol)


def _metodo_newton_modificado(x0, y0, itmax=20, tol=1e-6, raiz_tol=5e-2):
    return _newton_modificado_com_criterio_raiz(x0, y0, J, F, RAIZES_B, itmax=itmax, tol=tol, raiz_tol=raiz_tol)


def _metodo_newton_discreto(x0, y0, itmax=20, tol=1e-6, raiz_tol=5e-2):
    return _newton_com_criterio_raiz(x0, y0, J_aprox, F, RAIZES_B, itmax=itmax, tol=tol, raiz_tol=raiz_tol)


def _calcular_mapas(metodo, x_min, x_max, y_min, y_max, N=201, itmax=20, tol=1e-6, raiz_tol=5e-2):
    C_map = np.zeros((N + 1, N + 1), dtype=int)
    M_map = np.zeros((N + 1, N + 1), dtype=int)

    for i in range(N + 1):
        y0 = y_min + (y_max - y_min) * i / N
        for j in range(N + 1):
            x0 = x_min + (x_max - x_min) * j / N
            raiz, it, convergiu = metodo(x0, y0, itmax=itmax, tol=tol, raiz_tol=raiz_tol)
            if convergiu:
                C_map[i, j] = raiz
                M_map[i, j] = it

    return C_map, M_map


def sistemaB(N=500, itmax=20, tol=1e-6):
    """Calcula os mapas dos três métodos de Newton para o Sistema B."""
    resultados = {
        "metodos": {},
        "raizes": RAIZES_B,
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": -2.0,
        "y_max": 2.0,
        "N": N,
        "itmax": itmax,
        "tol": tol,
    }

    metodos = {
        "Newton": _metodo_newton,
        "Newton Modificado": _metodo_newton_modificado,
        "Newton Discreto": _metodo_newton_discreto,
    }

    for nome, metodo in metodos.items():
        C_map, M_map = _calcular_mapas(
            metodo,
            resultados["x_min"],
            resultados["x_max"],
            resultados["y_min"],
            resultados["y_max"],
            N=N,
            itmax=itmax,
            tol=tol,
            raiz_tol=5e-2,
        )
        resultados["metodos"][nome] = {
            "convergencia": C_map,
            "iteracoes": M_map,
        }

    return resultados

def main():
    print("Sistema B")
    sistemaB()

if __name__ == "__main__":
    main()
