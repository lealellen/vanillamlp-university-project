import numpy as np
import zipfile
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from network import MLP
from PIL import Image

# Descompactar o arquivo X_png.zip
zip_file = 'X_png.zip'
output_folder = 'images/'

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

# Caminho correto para a pasta descompactada
image_folder = os.path.join(output_folder, 'X_png')

# Carregar os dados dos pixels do arquivo X.txt
x_file = 'X.txt'
X = np.genfromtxt(x_file, delimiter=",")

# Carregar as letras correspondentes do arquivo Y_letra.txt
y_file = 'Y_letra.txt'
with open(y_file, 'r') as file:
    Y = file.read().splitlines()

# Converter as letras para valores numéricos
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Separar os dados em treino e teste (últimos 130 para teste)
X_train = X[:-130]
Y_train = Y_encoded[:-130]

X_test = X[-130:]
Y_test = Y_encoded[-130:]

# One-hot encoding para as saídas de treino
onehot_encoder = OneHotEncoder(sparse_output=False)
Y_train_reshaped = Y_train.reshape(-1, 1)
Y_train_onehot = onehot_encoder.fit_transform(Y_train_reshaped)

# Definir os parâmetros da rede neural
input_size = X_train.shape[1]  # Número de características (120 pixels)
hidden_layers = 3              # Número de neurônios na camada oculta (você pode ajustar)
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

# -------------------------------------------------------------------------
# OPCIONAL: Se você quiser carregar as imagens da pasta X_png:

def load_images_from_folder(folder, target_size=(32, 32)):
    images = []
    filenames = sorted(os.listdir(folder))  # Para garantir ordem
    for filename in filenames:
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # 'L' = grayscale (preto e branco)
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # Normalizar entre 0 e 1
            images.append(img_array.flatten())  # Achatar a imagem
    return np.array(images)

# Exemplo para carregar as imagens:
# images_data = load_images_from_folder(image_folder)
# print("Formato dos dados de imagem:", images_data.shape)
