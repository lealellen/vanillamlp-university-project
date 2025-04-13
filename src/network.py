import numpy as np
import pandas as pd

class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, epochs=1000):
        # Inicializar os parâmetros da rede (pesos, bias, etc.)
        pass

    def initialize_weights(self):
        # Inicializar pesos e biases com valores aleatórios
        pass

    def activation(self, z):
        # Função de ativação (ex: sigmoid, ReLU)
        pass

    def activation_derivative(self, z):
        # Derivada da função de ativação
        pass

    def forward(self, X):
        # Propagação para frente (forward pass)
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

    def fit(self, X, y):
        # Treinamento da rede neural (loop de epochs)
        pass

    def predict(self, X):
        # Realizar predições com a rede treinada
        pass

    def accuracy(self, y_true, y_pred):
        acuraccy = y_true/y_pred
        return acuraccy

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
 