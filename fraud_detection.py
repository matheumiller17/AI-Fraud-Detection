
import os
import pandas as pd
import joblib
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix
)

# 1. Download do dataset
print("Carregando dataset...")

caminho = kagglehub.dataset_download(
    "mlg-ulb/creditcardfraud"
)

arquivo = os.path.join(caminho, "creditcard.csv")
dados = pd.read_csv(arquivo)

# 2. Validar os dados
if len(dados) != 284807 or dados.isna().any().any():
    raise ValueError("Dataset incompleto ou com valores ausentes.")

print("Transacoes carregadas:", len(dados))

# 3. Separar variaveis
X = dados.drop("Class", axis=1)
y = dados["Class"].astype(int)

# 4. Treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Criar a IA
modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 6. Treinar modelo
print("Treinando Inteligencia Artificial...")
modelo.fit(X_treino, y_treino)

# 7. Avaliar
previsoes = modelo.predict(X_teste)
probabilidades = modelo.predict_proba(X_teste)[:, 1]

print("\nResultados:")
print(classification_report(y_teste, previsoes))

print(
    "Average Precision:",
    round(average_precision_score(
        y_teste, probabilidades
    ), 4)
)

print("\nMatriz de confusao:")
print(confusion_matrix(y_teste, previsoes))

# 8. Salvar modelo
joblib.dump(modelo, "fraud_detection_model.pkl")

print("\nModelo salvo com sucesso!")
