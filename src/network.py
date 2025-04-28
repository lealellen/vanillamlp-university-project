import numpy as np
import pandas as pd
import random
# Vamos usar o random para inicializar os pesos com valores aleatórios, o que garante que a rede não comece sempre do mesmo jeito

class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, epochs=1000):
        # Atribuindo os parâmetros como atributos da classe

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Inicializa os pesos e bias
        self.initialize_weights()

    def initialize_weights(self):
                
        self.in_weigths = [[random.uniform(-1, 1) for _ in range(self.hidden_layers)] for _ in range(self.input_size)]
        self.out_weigths = [[random.uniform(-1, 1) for _ in range(self.output_size)] for _ in range(self.hidden_layers)]
        self.hidden_bias = [random.uniform(-1, 1) for _ in range(self.hidden_layers)]
        self.out_bias = [random.uniform(-1, 1) for _ in range(self.output_size)]

    def activation(self, z):
        # Função de ativação (ex: sigmoid, ReLU)
        pass

    def activation_derivative(self, z):
        """Função para calculo da derivada da sigmoide que é usada no cálculo do gradiente durante o backpropagation.
            Ela permite ajustar os pesos da rede neural com base no erro da previsão.
        """
        pass

    def forward(self, X):
        # Propagação para frente (forward pass)
        pass

    def fit(self, X, y):
        # Treinamento da rede neural (loop de epochs)
        pass

    def predict(self, X):
        # Realizar predições com a rede treinada
        pass

    def compute_loss(self, y_true, y_pred):
        # Calcular a função de perda (ex: MSE, cross-entropy)
        pass

    def backward(self, X, y_true, y_pred):
        # Propagação para trás (backward pass) e cálculo dos gradientes
        pass

    def update_weights(self, gradients):
        # Atualizar os pesos usando os gradientes calculados
        pass

    def accuracy(self, y_true, y_pred):
        """
        Realiza o calculo da acurácia
        """
        correct_predictions = np.sum(y_true == y_pred)
        accuracy = correct_predictions / len(y_true)
        return accuracy

    def final_report():
        """
        Gera arquivos contendo as seguintes informações:
            Hiperparâmetros finais da arquitetura da rede neural e hiperparâmetros de inicialização.
            Pesos iniciais da rede.
            Pesos finais da rede.
            Erro cometido pela rede neural em cada iteração do treinamento.
            Saídas produzidas pela rede neural para cada um dos dados de teste realizados
        """
        pass
 