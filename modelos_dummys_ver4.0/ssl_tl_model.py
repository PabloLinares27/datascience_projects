import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

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

# Si las etiquetas son texto, mapearlas a números
mapping = {'buy': 0, 'sell': 1, 'hold': 2}
df[target] = df[target].map(mapping)

# Separar variables predictoras y objetivo
X = df[features]
y = df[target]

# Dividir en conjunto de entrenamiento y prueba (con estratificación)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Aplicar PCA para obtener representaciones latentes (reducción de dimensionalidad)
pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Definir y entrenar el modelo XGBoost
clf = XGBClassifier(n_estimators=500, random_state=42, objective='multi:softmax', num_class=3)
clf.fit(X_train_pca, y_train)