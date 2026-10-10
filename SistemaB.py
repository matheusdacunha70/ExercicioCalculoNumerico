import numpy as np

from newtons import newton, newton_discreto, newton_modificado


def F(x, y):
    F = np.zeros(2)
    F[0] = x**5 - 10*x**3 * y**2 + 5*x*y**4 - 1
    F[1] = 5 * x**4 * y - 10*x**2 * y**3 - y**5
    return F

def J(x, y):
    J = np.zeros((2,2))
    J[0][0] = 5*x**4 - 30*x**2 * y**2 + 5*y**4
    J[0][1] = 20 * x**3 * y + 20*x*y**3
    J[1][0] = 20 * x**3*y - 20*x * y**3
    J[1][1] = 5 * x**4 - 30*x**2 * y**2 - 5*y**4
    return J

def sistemaB():
    N = 500
    itmax = 20
    tol = 1e-6

    for i in range(0, N+1):
        for j in range(0, N+1):
            x = -2.0 + 4*i/N
            y = -2.0 + 4*j/N

            try:
                x, y, it, erro = newton(x, y, J, F, itmax=itmax, tol=tol)
                #print("Iteração: ", it, "Erro: ", erro, "x: ", x, "y: ", y, "Para uma aproximação inicial de x: ", -1.5 + 3*i/N, "y: ", -1.5 + 3*j/N)
                x, y, it, erro = newton_modificado(x, y, J, F, itmax=itmax, tol=tol)
                #print("Iteração: ", it, "Erro: ", erro, "x: ", x, "y: ", y)
            except np.linalg.LinAlgError:
                print("Ponto inicial singular: x: ", x, "y: ", y)

def main():
    print("Sistema B")
    sistemaB()

if __name__ == "__main__":
    main()
