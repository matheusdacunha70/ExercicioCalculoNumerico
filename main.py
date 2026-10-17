import matplotlib.pyplot as plt

from gerar_grafico import grafico_calor
from SistemaA import sistemaA
from SistemaB import sistemaB


def main():
    resultados = sistemaA(500, 20, 1e-6)
    grafico_calor(resultados, titulo="Sistema A")
    plt.show()

    resultados = sistemaB(500, 20, 1e-6)
    grafico_calor(resultados, titulo="Sistema B")
    plt.show()


if __name__ == "__main__":
    main()
