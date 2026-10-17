import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def _mapa_cores_raizes(raizes):
    cores = [
        "#f0f0f0",
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ]

    limites = np.arange(-0.5, len(raizes) + 1.5, 1)
    return ListedColormap(cores[: len(raizes) + 1]), BoundaryNorm(limites, len(cores[: len(raizes) + 1]))


def grafico_calor(resultados, titulo=None):
    """Plota os mapas de convergência e iterações já calculados."""
    metodos = resultados["metodos"]
    raizes = resultados["raizes"]
    x_min = resultados["x_min"]
    x_max = resultados["x_max"]
    y_min = resultados["y_min"]
    y_max = resultados["y_max"]

    cmap_raizes, norm_raizes = _mapa_cores_raizes(raizes)

    fig, axes = plt.subplots(len(metodos), 2, figsize=(14, 5 * len(metodos)), squeeze=False)

    for linha, (nome, dados) in enumerate(metodos.items()):
        im1 = axes[linha, 0].imshow(
            dados["convergencia"],
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap=cmap_raizes,
            norm=norm_raizes,
            aspect="auto",
        )
        axes[linha, 0].set_title(f"{nome} - Região de Convergência")
        axes[linha, 0].set_xlabel("x")
        axes[linha, 0].set_ylabel("y")
        ticks = np.arange(0, len(raizes) + 1)
        labels = ["C0"] + [f"R{i}: ({raiz[0]:.2f}, {raiz[1]:.2f})" for i, raiz in enumerate(raizes, start=1)]
        cbar1 = fig.colorbar(im1, ax=axes[linha, 0], ticks=ticks, label="Raiz")
        cbar1.ax.set_yticklabels(labels)

        im2 = axes[linha, 1].imshow(
            dados["iteracoes"],
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap="plasma",
            aspect="auto",
        )
        axes[linha, 1].set_title(f"{nome} - Número de Iterações")
        axes[linha, 1].set_xlabel("x")
        axes[linha, 1].set_ylabel("y")
        fig.colorbar(im2, ax=axes[linha, 1], label="Iterações (k)")

    if titulo:
        fig.suptitle(titulo, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    return fig, axes
