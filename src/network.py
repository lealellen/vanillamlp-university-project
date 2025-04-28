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
        self.in_weights = np.random.uniform(-1, 1, size=(self.input_size, self.hidden_layers))
        self.out_weights = np.random.uniform(-1, 1, size=(self.hidden_layers, self.output_size))
        self.in_bias = np.random.uniform(-1, 1, size=(1, self.hidden_layers))
        self.out_bias = np.random.uniform(-1, 1, size=(1, self.output_size))
        
    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    def activation_derivative(self, z):
        sig = self.activation(z)
        return sig * (1 - sig)

    def forward(self, X):
        hidden_input = X @ self.in_weights + self.in_bias
        hidden_output = self.activation(hidden_input)
        final_input = hidden_output @ self.out_weights + self.out_bias
        y_pred = self.activation(final_input)

        return y_pred, final_input, hidden_input, hidden_output
    
    def update_weights(self, delta_oculta, delta_saida, X, hidden_output):
        # Atualiza os pesos da camada de saída: produto da transposta da saída da camada oculta com o delta da camada de saída, multiplicado pela taxa de aprendizado.
        self.out_weights += self.learning_rate * (hidden_output.T @ delta_saida)
        # Atualiza o bias da camada de saída: soma do delta da camada de saída, multiplicado pela taxa de aprendizado.
        self.out_bias += self.learning_rate * np.sum(delta_saida, axis=0, keepdims=True)
        # Atualiza os pesos da camada oculta: produto da transposta da entrada X com o delta da camada oculta, multiplicado pela taxa de aprendizado.
        self.in_weights += self.learning_rate * (X.T @ delta_oculta)
        # Atualiza o bias da camada oculta: soma do delta da camada oculta, multiplicado pela taxa de aprendizado.
        self.in_bias += self.learning_rate * np.sum(delta_oculta, axis=0, keepdims=True)

    def fit(self, X, y):
        errors = []  # Salva o erro por época
        for epoch in range(self.epochs):
            # Realiza a propagação para frente (forward pass) e calcula a previsão (y_pred), entradas e saídas das camadas.
            y_pred, final_input, hidden_input, hidden_output = self.forward(X)
            # Calcula o erro entre as previsões (y_pred) e os valores reais (y).
            erro = y - y_pred
            # Calcula o delta da camada de saída: erro multiplicado pela derivada da função de ativação da camada de saída.
            delta_saida = erro * self.activation_derivative(final_input)
            # Calcula o erro da camada oculta: multiplicação do delta da saída pelos pesos de saída transpostos.
            erro_oculto = delta_saida @ self.out_weights.T
            # Calcula o delta da camada oculta: erro oculto multiplicado pela derivada da função de ativação da camada oculta.
            delta_oculta = erro_oculto * self.activation_derivative(hidden_input)

            self.update_weights(delta_oculta, delta_saida, X, hidden_output)

            # Calcular o erro total e adicionar ao histórico
            loss = np.mean(np.square(erro))  # Exemplo de MSE
            errors.append(loss)
            if epoch % 100 == 0:  # Exibe a cada 100 épocas
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss}")
        
        return errors
                
    def predict(self, X):
        y_pred, _, _, _ = self.forward(X)
        return y_pred

    def compute_loss(self, y_true, y_pred):
        # Calcular a função de perda (ex: MSE, cross-entropy)
        pass

    def backward(self, X, y_true, y_pred):
        # Propagação para trás (backward pass) e cálculo dos gradientes
        pass

    def accuracy(self, y_true, y_pred):
        """
        Realiza o calculo da acurácia
        """
        correct_predictions = np.sum(y_true == y_pred)
        accuracy = correct_predictions / len(y_true)
        return accuracy

    def final_report(self, errors, file_name="final_report.txt"):
        """
        Gera arquivos contendo as seguintes informações:
            Hiperparâmetros finais da arquitetura da rede neural e hiperparâmetros de inicialização.
            Pesos iniciais da rede.
            Pesos finais da rede.
            Erro cometido pela rede neural em cada iteração do treinamento.
            Saídas produzidas pela rede neural para cada um dos dados de teste realizados
        """
        with open(file_name, "w") as f:
            f.write(f"Final Report - MLP\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Learning Rate: {self.learning_rate}\n")
            f.write(f"Input Size: {self.input_size}\n")
            f.write(f"Hidden Layers: {self.hidden_layers}\n")
            f.write(f"Output Size: {self.output_size}\n")
            f.write("\nWeights (initial):\n")
            f.write(f"{self.in_weights}\n")
            f.write(f"{self.out_weights}\n")
            f.write("\nBiases (initial):\n")
            f.write(f"{self.in_bias}\n")
            f.write(f"{self.out_bias}\n")
            f.write("\nLoss per Epoch:\n")
            for epoch, error in enumerate(errors):
                f.write(f"Epoch {epoch + 1}: Loss = {error}\n")
            f.write("\nFinal Weights:\n")
            f.write(f"{self.in_weights}\n")
            f.write(f"{self.out_weights}\n")
    