import numpy as np
import matplotlib.pyplot as plt


def _calcular_mapas(F, J, roots, x_min, x_max, y_min, y_max, N=500, k_max=20, epsilon=1e-6):
    """Calcula os mapas de convergência e de iterações para um sistema F(x, y)."""
    x_vals = np.linspace(x_min, x_max, N + 1)
    y_vals = np.linspace(y_min, y_max, N + 1)

    C_map = np.zeros((len(y_vals), len(x_vals)), dtype=int)
    M_map = np.zeros((len(y_vals), len(x_vals)), dtype=int)

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            x_k = np.array([x, y], dtype=float)
            convergiu = False
            iter_count = 0

            for _ in range(k_max):
                fx = F(x_k[0], x_k[1])

                if np.linalg.norm(fx) < epsilon:
                    break

                try:
                    jac = J(x_k[0], x_k[1])
                    delta = np.linalg.solve(jac, -fx)
                except np.linalg.LinAlgError:
                    break

                x_k = x_k + delta
                iter_count += 1

                for idx, root in enumerate(roots):
                    if np.linalg.norm(x_k - root) < epsilon:
                        C_map[i, j] = idx + 1
                        M_map[i, j] = iter_count
                        convergiu = True
                        break

                if convergiu:
                    break

            if not convergiu:
                C_map[i, j] = 0
                M_map[i, j] = 0

    return C_map, M_map, x_vals, y_vals


def grafico_convergencia(F, J, roots, x_min, x_max, y_min, y_max, N=500, k_max=20, epsilon=1e-6, cmap="viridis"):
    """Gera o mapa de calor para a região de convergência do sistema."""
    C_map, _, _, _ = _calcular_mapas(F, J, roots, x_min, x_max, y_min, y_max, N, k_max, epsilon)

    fig, ax = plt.subplots(figsize=(7, 6))
    imagem = ax.imshow(
        C_map,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )
    ax.set_title("Mapa de Convergência")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(imagem, ax=ax, label="Raiz / C_0")
    plt.tight_layout()
    return fig, ax


def grafico_iteracoes(F, J, roots, x_min, x_max, y_min, y_max, N=500, k_max=20, epsilon=1e-6, cmap="plasma"):
    """Gera o mapa de calor para o número de iterações até a convergência."""
    _, M_map, _, _ = _calcular_mapas(F, J, roots, x_min, x_max, y_min, y_max, N, k_max, epsilon)

    fig, ax = plt.subplots(figsize=(7, 6))
    imagem = ax.imshow(
        M_map,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )
    ax.set_title("Mapa de Iterações")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(imagem, ax=ax, label="Iterações (k)")
    plt.tight_layout()
    return fig, ax


def grafico_calor(F, J, roots, x_min, x_max, y_min, y_max, N=500, k_max=20, epsilon=1e-6):
    """Cria os dois gráficos de calor em uma única figura."""
    C_map, M_map, _, _ = _calcular_mapas(F, J, roots, x_min, x_max, y_min, y_max, N, k_max, epsilon)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im1 = axes[0].imshow(
        C_map,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    axes[0].set_title("Região de Convergência")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im1, ax=axes[0], label="Raiz / C_0")

    im2 = axes[1].imshow(
        M_map,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="plasma",
        aspect="auto",
    )
    axes[1].set_title("Número de Iterações")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im2, ax=axes[1], label="Iterações (k)")

    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    # Exemplo de uso com o Sistema A.
    from SistemaA import F, J

    raizes = [
        np.array([1.0, 0.0]),
        np.array([-0.5, np.sqrt(3) / 2]),
        np.array([-0.5, -np.sqrt(3) / 2]),
    ]

    grafico_calor(F, J, raizes, -1.5, 1.5, -1.5, 1.5, N=200)
    plt.show()