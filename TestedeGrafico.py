import matplotlib.pyplot as plt
import numpy as np


# 1. Definir o sistema F(x, y) e sua Matriz Jacobiana J(x, y) para o item (b)
def F(x, y):
    f1 = x**5 - 10 * (x**3) * (y**2) + 5 * x * (y**4) - 1
    f2 = 5 * (x**4) * y - 10 * (x**2) * (y**3) - y**5
    return np.array([f1, f2])


def jacobian(x, y):
    # Derivadas parciais para a matriz Jacobiana J(x, y)
    df1_dx = 5 * (x**4) - 30 * (x**2) * (y**2) + 5 * (y**4)
    df1_dy = -20 * (x**3) * y + 20 * x * (y**3)

    df2_dx = 20 * (x**3) * y - 20 * x * (y**3)
    df2_dy = 5 * (x**4) - 30 * (x**2) * (y**2) - 5 * (y**4)
    return np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])


# 2. As 5 soluções (raízes) fornecidas no enunciado
roots = [
    np.array([1.0, 0.0]),  # (x1*, y1*) = (1, 0)
    np.array(
        [np.cos(2 * np.pi / 5), np.sin(2 * np.pi / 5)]
    ),  # (x2*, y2*) = (cos(2pi/5), sin(2pi/5))
    np.array(
        [np.cos(4 * np.pi / 5), np.sin(4 * np.pi / 5)]
    ),  # (x3*, y3*) = (cos(4pi/5), sin(4pi/5))
    np.array(
        [np.cos(6 * np.pi / 5), np.sin(6 * np.pi / 5)]
    ),  # (x4*, y4*) = (cos(6pi/5), sin(6pi/5))
    np.array(
        [np.cos(8 * np.pi / 5), np.sin(8 * np.pi / 5)]
    ),  # (x5*, y5*) = (cos(8pi/5), sin(8pi/5))
]

# 3. Parâmetros da região T e do método
N = 500  # N >= 500 (aumente para 500 se quiser mais definição)
k_max = 20  # Número máximo de iterações
epsilon = 1e-6  # Tolerância

# Região T: -2 a 2
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0

x_vals = np.linspace(x_min, x_max, N + 1)
y_vals = np.linspace(y_min, y_max, N + 1)

# Matrizes de mapeamento
C_map = np.zeros((len(y_vals), len(x_vals)), dtype=int)
M_map = np.zeros((len(y_vals), len(x_vals)), dtype=int)

# 4. Aplicar o Método de Newton para cada ponto da grade T
for i, y in enumerate(y_vals):
    for j, x in enumerate(x_vals):
        x_k = np.array([x, y], dtype=float)
        converged = False
        iter_count = 0

        for k in range(k_max):
            fx = F(x_k[0], x_k[1])

            if np.linalg.norm(fx) < epsilon:
                break

            J = jacobian(x_k[0], x_k[1])

            try:
                delta = np.linalg.solve(J, -fx)
            except np.linalg.LinAlgError:
                break

            x_k = x_k + delta
            iter_count += 1

            # Verificar convergência para uma das 5 raízes
            r_index = -1
            for idx, root in enumerate(roots):
                if np.linalg.norm(x_k - root) < epsilon:
                    r_index = idx + 1  # 1 a 5 para as raízes
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

# 5. Plotagem dos Gráficos com Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Primeiro mapa: Região de convergência por raiz (1 a 5, e 0 para C_0)
im1 = axes[0].imshow(
    C_map,
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    cmap="tab10",
)
axes[0].set_title("Fractal de Newton - Região de Convergência (5 Raízes)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(im1, ax=axes[0], label="Raiz (1-5) / C_0")

# Segundo mapa: Número de iterações
im2 = axes[1].imshow(
    M_map, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="plasma"
)
axes[1].set_title("Número de Iterações (k)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(im2, ax=axes[1], label="Iterações (k)")

plt.tight_layout()
plt.show()