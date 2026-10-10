import numpy as np
import matplotlib.pyplot as plt

from gerar_grafico import grafico_calor
from SistemaA import F as F_A, J as J_A
from SistemaB import F as F_B, J as J_B


def main():
    raizes_a = [
        np.array([1.0, 0.0]),
        np.array([-0.5, np.sqrt(3) / 2]),
        np.array([-0.5, -np.sqrt(3) / 2]),
    ]

    raizes_b = [
        np.array([1.0, 0.0]),
        np.array([np.cos(2 * np.pi / 5), np.sin(2 * np.pi / 5)]),
        np.array([np.cos(4 * np.pi / 5), np.sin(4 * np.pi / 5)]),
        np.array([np.cos(6 * np.pi / 5), np.sin(6 * np.pi / 5)]),
        np.array([np.cos(8 * np.pi / 5), np.sin(8 * np.pi / 5)]),
    ]

    fig_a, axes_a = grafico_calor(F_A, J_A, raizes_a, -1.5, 1.5, -1.5, 1.5, N=200)
    fig_a.suptitle("Sistema A", fontsize=14)
    plt.show()

    fig_b, axes_b = grafico_calor(F_B, J_B, raizes_b, -2.0, 2.0, -2.0, 2.0, N=200)
    fig_b.suptitle("Sistema B", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
