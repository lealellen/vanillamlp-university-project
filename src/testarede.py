import numpy as np
import zipfile
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from network import MLP 

# Descompactar o arquivo X_png.zip
zip_file = 'X_png.zip'
output_folder = 'images/'

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

# Carregar os dados dos pixels do arquivo X.txt
x_file = 'X.txt'
X = np.genfromtxt(x_file, delimiter=",") 

# Carregar as letras correspondentes do arquivo Y_letra.txt
y_file = 'Y_letra.txt'
Y = np.loadtxt(y_file, dtype=str)

# Converter as letras para valores numéricos
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Separar os dados em treino e teste
X_train = X[:-130]
Y_train = Y_encoded[:-130]

X_test = X[-130:]
Y_test = Y_encoded[-130:]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y_train)

# Agora faz one-hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y_train_onehot = onehot_encoder.fit_transform(integer_encoded)

# Definir os parâmetros da rede neural
input_size = X_train.shape[1]  # Número de características (120 pixels)
hidden_layers = 3  # Número de neurônios na camada oculta
output_size = len(np.unique(Y_encoded))  # Número de classes (letras diferentes)

# Instanciar o modelo MLP
mlp = MLP(input_size, hidden_layers, output_size, learning_rate=0.01, epochs=50)

# Treinar o modelo
print("Iniciando o treinamento...")
errors = mlp.fit(X_train, Y_train_onehot)

# Avaliar o modelo
y_pred = mlp.predict(X_test)
accuracy = mlp.accuracy(Y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

# Gerar o relatório final
mlp.final_report(errors, file_name="final_report.txt")
print("Treinamento concluído e relatório gerado.")
