import numpy as np

from newtons import newton, newton_modificado


def F(x,y):
    F = np.zeros(2)
    F[0] = x**3 -3*x*y**2 -1    #f1(x,y)
    F[1] = 3*x**2*y - y**3      #f2(x,y)
    return F

def J(x,y):
    J = np.zeros((2,2))
    J[0][0] = 3*x**2 - 3*y**2       #df1/dx
    J[0][1] = -6*x*y                #df1/dy
    J[1][0] = 6*x*y                 #df2/dx
    J[1][1] = 3*x**2 - 3*y**2       #df2/dy
    return J

def J_aprox(x,y):
    J = np.zeros((2,2))
    h = 1e-6
    J[0][0] = (F(x+h,y)[0] - F(x,y)[0])/h
    J[0][1] = (F(x,y+h)[0] - F(x,y)[0])/h
    J[1][0] = (F(x+h,y)[1] - F(x,y)[1])/h
    J[1][1] = (F(x,y+h)[1] - F(x,y)[1])/h
    return J

def sistemaA():
    N = 201
    itmax = 20
    tol = 1e-6

    for i in range(0, N+1):
        for j in range(0, N+1):
            x = -1.5 + 3*i/N
            y = -1.5 + 3*j/N
            try:
                x, y, it, erro = newton(x, y, J, F, itmax=itmax, tol=tol)
                #print("Iteração: ", it, "Erro: ", erro, "x: ", x, "y: ", y, "Para uma aproximação inicial de x: ", -1.5 + 3*i/N, "y: ", -1.5 + 3*j/N)
                #x, y, it, erro = newton_modificado(x, y, J, F, itmax=itmax, tol=tol)
                #print("Iteração: ", it, "Erro: ", erro, "x: ", x, "y: ", y)
                #x, y, it, erro = newton(x, y, J_aprox, F, itmax=itmax, tol=tol) #Newton Discreto
            except np.linalg.LinAlgError:
                print("Ponto inicial singular: x: ", x, "y: ", y)



def main():
    print("Sistema A")
    sistemaA()
    print("acabou")


if __name__ == "__main__":
    print("Resolvendo sistemas não lineares")
    main()
