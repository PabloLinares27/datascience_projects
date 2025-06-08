import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

os.chdir('C:/Users/POZOLE/Documents/pp/nohtyp/practice_algo_trading/modelos_dummys_ver2.0/data_modelos_dummys2.0')
data = pd.read_csv('modelo_supervisado_2.0.csv')

# Cargar los datos
df = data.copy()

# Definir las columnas predictoras y la variable objetivo
features = ['open','value','rsi', 'macd', 'adx', 'adp', 'adm', 'dri',
            'dlogri', 'avgtr', 'donch_hband', 'donch_lband', 'vwap']

target = 'signals'

# Eliminar valores nulos
df = df.dropna()

# Mapear etiquetas de texto a numéricas (si es necesario)
mapping = {'buy': 0, 'sell': 1, 'hold': 2}
df[target] = df[target].map(mapping)

# Separar variables predictoras y objetivo
X = df[features]
y = df[target]

# Dividir en conjunto de entrenamiento y prueba (estratificado)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir los datos a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Crear DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definir un modelo FT-Transformer simple para datos tabulares
class FTTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_token=32, n_heads=4, depth=3, mlp_ratio=4):
        super(FTTransformerClassifier, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_token)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.Linear(d_token, d_token * mlp_ratio),
            nn.ReLU(),
            nn.Linear(d_token * mlp_ratio, num_classes)
        )
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.input_layer(x)  # (batch_size, d_token)
        # Los módulos Transformer esperan entrada de forma (seq_len, batch_size, d_model)
        # En datos tabulares, podemos tratar cada muestra como una secuencia de longitud 1.
        x = x.unsqueeze(0)  # (1, batch_size, d_token)
        x = self.transformer_encoder(x)  # (1, batch_size, d_token)
        x = x.squeeze(0)  # (batch_size, d_token)
        logits = self.classifier(x)  # (batch_size, num_classes)
        return logits

# Instanciar el modelo
input_dim = X_train.shape[1]
num_classes = len(np.unique(y))
model = FTTransformerClassifier(input_dim=input_dim, num_classes=num_classes)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Entrenar el modelo FT-Transformer
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluar el modelo en el conjunto de prueba
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.numpy())
        all_targets.append(batch_y.numpy())

y_pred_ft = np.concatenate(all_preds)
y_test_ft = np.concatenate(all_targets)

