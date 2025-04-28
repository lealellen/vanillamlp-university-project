import pytest
from network import MLP
import numpy as np

# Teste 1: Inicialização dos pesos e biases
def test_inicializacao_pesos_biases():
    mlp = MLP(input_size=3, hidden_layers=3, output_size=2, learning_rate=0.01, epochs=1000)

    # Verificar se os pesos de entrada e saída são inicializados corretamente
    print("Pesos de entrada:", mlp.in_weigths)
    print("Pesos de saída:", mlp.out_weigths)
    print("Bias da camada escondida:", mlp.hidden_bias)
    print("Bias da camada de saída:", mlp.out_bias)

    # Testar se os valores estão dentro do intervalo esperado (-1, 1)
    for weights in mlp.in_weigths:
        for w in weights:
            assert -1 <= w <= 1, f"Peso de entrada fora do intervalo: {w}"

    for weights in mlp.out_weigths:
        for w in weights:
            assert -1 <= w <= 1, f"Peso de saída fora do intervalo: {w}"

    for bias in mlp.hidden_bias:
        assert -1 <= bias <= 1, f"Bias da camada escondida fora do intervalo: {bias}"

    for bias in mlp.out_bias:
        assert -1 <= bias <= 1, f"Bias da camada de saída fora do intervalo: {bias}"

    print("Testes de inicialização passaram!")

# Teste 2: Cálculo da acurácia
def test_calculo_acuracia():
    y_true = np.array([0, 1, 1, 0])  # Verdadeiras classes
    y_pred = np.array([0, 1, 0, 0])  # Predições do modelo

    # Calcular a acurácia
    mlp = MLP(input_size=3, hidden_layers=3, output_size=2, learning_rate=0.01, epochs=1000)
    accuracy = mlp.accuracy(y_true, y_pred)

    # A acurácia esperada é 3 acertos em 4 previsões
    expected_accuracy = 3 / 4
    print(f"Acurácia calculada: {accuracy * 100}%")

    # Comparar a acurácia calculada com a esperada
    assert np.isclose(accuracy, expected_accuracy), f"Acurácia esperada {expected_accuracy}, mas obtivemos {accuracy}"

    print("Teste de acurácia passou!")

# Teste 3: Testar os parâmetros de inicialização
def test_parametros_inicializacao():
    mlp = MLP(input_size=3, hidden_layers=3, output_size=2, learning_rate=0.05, epochs=2000)

    # Verificar se os parâmetros de inicialização estão corretos
    print(f"Épocas: {mlp.epochs}, Learning rate: {mlp.learning_rate}")

    # Validar com assertivas
    assert mlp.epochs == 2000, f"Valor de épocas esperado 2000, mas obtivemos {mlp.epochs}"
    assert mlp.learning_rate == 0.05, f"Valor de learning_rate esperado 0.05, mas obtivemos {mlp.learning_rate}"

    print("Testes de inicialização passaram!")

# Teste 4: Teste geral da classe MLP
def test_geral_mlp():
    mlp = MLP(input_size=3, hidden_layers=3, output_size=2, learning_rate=0.05, epochs=2000)

    # Verificar se a estrutura da rede foi inicializada corretamente
    print(f"Input size: {mlp.input_size}")
    print(f"Hidden layers: {mlp.hidden_layers}")
    print(f"Output size: {mlp.output_size}")
    print(f"Learning rate: {mlp.learning_rate}")
    print(f"Epochs: {mlp.epochs}")

    # Validar com assertivas
    assert mlp.input_size == 3, f"Input size esperado 3, mas obtivemos {mlp.input_size}"
    assert mlp.hidden_layers == 3, f"Hidden layers esperado 3, mas obtivemos {mlp.hidden_layers}"
    assert mlp.output_size == 2, f"Output size esperado 2, mas obtivemos {mlp.output_size}"
    assert mlp.learning_rate == 0.05, f"Learning rate esperado 0.05, mas obtivemos {mlp.learning_rate}"
    assert mlp.epochs == 2000, f"Epochs esperado 2000, mas obtivemos {mlp.epochs}"

    print("Testes gerais passaram!")

# Para rodar os testes, use o comando:
# pytest -v
