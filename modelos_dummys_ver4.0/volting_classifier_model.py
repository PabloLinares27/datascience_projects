import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

# Separar variables predictoras y objetivo
X = df[features]
y = df[target]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir clasificadores base
clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2 = LogisticRegression(max_iter=1000, random_state=42)
clf3 = SVC(kernel='linear', random_state=42)

# Crear el modelo de ensamblaje (VotingClassifier)
ensemble_clf = VotingClassifier(estimators=[
    ('rf', clf1), 
    ('lr', clf2), 
    ('svc', clf3)
], voting='hard')  # 'hard' vota por la clase más frecuente

# Entrenar el modelo de ensamblaje
ensemble_clf.fit(X_train, y_train)
