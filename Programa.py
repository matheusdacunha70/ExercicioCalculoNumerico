import matplotlib.pyplot as plt
import numpy as np

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

N = 50
itmax = 20
tol = 10e-6

for i in range(0, N+1):
    for j in range(0, N+1):
        it = 1
        erro = 1
        x = -1.5 + 3*i/N
        y = -1.5 + 3*j/N
        while(erro > tol and it < itmax):
            s = np.linalg.solve(J(x,y), -F(x,y))
            x = x + s[0]
            y = y + s[1]
            erro = np.linalg.norm(s)
            it += 1
        print("Iteração: ", it, "Erro: ", erro, "x: ", x, "y: ", y, "Para uma aproximação inicial de x: ", -1.5 + 3*i/N, "y: ", -1.5 + 3*j/N)


