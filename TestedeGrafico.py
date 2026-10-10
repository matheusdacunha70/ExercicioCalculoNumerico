import matplotlib.pyplot as plt
import numpy as np


# 1. Definir o sistema F(x, y) e sua Matriz Jacobiana J(x, y)
# Exemplo: F(x, y) = [x^3 - 3xy^2 - 1, 3x^2y - y^3] (equivalente a z^3 - 1 = 0)
def F(x, y):
    f1 = x**3 - 3 * x * (y**2) - 1
    f2 = 3 * (x**2) * y - y**3
    return np.array([f1, f2])


def jacobian(x, y):
    # Derivadas parciais para a matriz Jacobiana J(x, y)
    df1_dx = 3 * (x**2) - 3 * (y**2)
    df1_dy = -6 * x * y
    df2_dx = 6 * x * y
    df2_dy = 3 * (x**2) - 3 * (y**2)
    return np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])


# Raizes conhecidas do sistema para comparação (ex: z^3 - 1 = 0 tem 3 raízes reais/complexas)
# Raiz 1: (1, 0), Raiz 2: (-0.5, sqrt(3)/2), Raiz 3: (-0.5, -sqrt(3)/2)
roots = [
    np.array([1.0, 0.0]),
    np.array([-0.5, np.sqrt(3) / 2]),
    np.array([-0.5, -np.sqrt(3) / 2]),
]

# Parâmetros do método
k_max = 20  # Número máximo de iterações
epsilon = 10e-6  # Tolerância
N = 500  # Resolução da grade (N x N)

# Definir a região do plano T = {(xi, yj)}
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5
x_vals = np.linspace(x_min, x_max, N+1)
y_vals = np.linspace(y_min, y_max, N+1)

# Matrizes para armazenar os resultados
# C_map guardará o índice da raiz (0, 1, 2...) ou -1 para não convergência (C_0)
# M_map guardará o número de iterações ou 0
C_map = np.zeros((len(y_vals), len(x_vals)), dtype=int)
M_map = np.zeros((len(y_vals), len(x_vals)), dtype=int)

# 2. Aplicar o Método de Newton para cada ponto da grade T
for i, y in enumerate(y_vals):
    for j, x in enumerate(x_vals):
        x_k = np.array([x, y], dtype=float)
        converged = False
        iter_count = 0

        for k in range(k_max):
            fx = F(x_k[0], x_k[1])

            # Critério de parada preliminar pelo valor da função
            if np.linalg.norm(fx) < epsilon:
                break

            J = jacobian(x_k[0], x_k[1])

            try:
                # Resolve J * delta = -F para encontrar o passo de Newton
                delta = np.linalg.solve(J, -fx)
            except np.linalg.LinAlgError:
                break  # Jacobiana singular

            x_k = x_k + delta
            iter_count += 1

            # Verificar convergência para alguma das raízes conhecidas
            r_index = -1
            for idx, root in enumerate(roots):
                if np.linalg.norm(x_k - root) < epsilon:
                    r_index = idx + 1  # 1, 2, 3 para as raízes
                    converged = True
                    break
            if converged:
                break

        if converged:
            C_map[i, j] = r_index
            M_map[i, j] = iter_count
        else:
            C_map[i, j] = 0  # C_0 (não convergiu)
            M_map[i, j] = 0

# 3. Plotagem dos Gráficos com Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Primeiro mapa: Região de convergência (qual raiz convergiu)
im1 = axes[0].imshow(
    C_map,
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    cmap="viridis",
)
axes[0].set_title("Primeiro Mapa: Região de Convergência (por Raiz)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(im1, ax=axes[0], label="Raiz / C_0")

# Segundo mapa: Número de iterações realizadas
im2 = axes[1].imshow(
    M_map, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="plasma"
)
axes[1].set_title("Segundo Mapa: Número de Iterações (k)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(im2, ax=axes[1], label="Iterações (k)")

plt.tight_layout()
plt.show()