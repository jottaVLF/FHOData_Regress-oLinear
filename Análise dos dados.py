#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Função para limpar e converter os dados
def clean_and_convert_value(x):
    """
    Função robusta para limpar e converter valores string em números.
    Lida com faixas de valor (ex: '100-200'), divisões (ex: '1600/13.8') e outros caracteres.
    """
    s = str(x).strip().replace(' ', '')
    
    # Lida com casos de faixas de valor
    if '-' in s and s.count('-') == 1:
        try:
            parts = [float(p) for p in s.split('-') if p.strip()]
            return np.mean(parts)
        except (ValueError, IndexError):
            return np.nan
    
    # Lida com casos de divisões de valor
    elif '/' in s:
        try:
            parts = [float(p) for p in s.split('/') if p.strip()]
            if len(parts) == 2:
                return parts[0] / parts[1]
            else:
                return np.nan
        except (ValueError, IndexError, ZeroDivisionError):
            return np.nan
            
    # Lida com valores numéricos simples
    else:
        try:
            return float(s)
        except ValueError:
            return np.nan

# 1. Limpeza dos Dados
# --------------------------------------------------------------------------------
# Carrega o arquivo com a codificação correta
try:
    df = pd.read_csv("Cars Datasets 2025.csv", encoding='latin-1')
except UnicodeDecodeError:
    df = pd.read_csv("Cars Datasets 2025.csv", encoding='iso-8859-1')

print("Dados originais:\n")
print(df.head())
print("\n" + "="*50 + "\n")

# Dicionário de limpeza para múltiplas colunas
cleaning_map = {
    ' cc': '',
    'kwh': '',
    'hp': '',
    ' km/h': '',
    'sec': '',
    'Nm': '',
    '$': '',
    ',': ''
}

# Limpa e converte as colunas para numérico
for col in ['CC/Battery Capacity', 'HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H', 'Torque', 'Cars Prices']:
    if col in df.columns:
        # Substitui os valores com base no dicionário
        for k, v in cleaning_map.items():
            df[col] = df[col].astype(str).str.replace(k, v, regex=False)
        
        # Aplica a função de limpeza e conversão
        df[col] = df[col].apply(clean_and_convert_value)

# Remove linhas com valores nulos
df.dropna(inplace=True)

print("Dados após a limpeza:\n")
print(df.head())
print("\n" + "="*50 + "\n")

# 2. Matriz de Correlação
# --------------------------------------------------------------------------------
# Seleciona apenas as colunas numéricas para a correlação
numeric_cols = ['CC/Battery Capacity', 'HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H', 'Torque', 'Cars Prices']
correlation_matrix = df[numeric_cols].corr()

# Cria o mapa de calor da matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação das Características dos Carros')
plt.show()

# 3. Regressão Linear
# --------------------------------------------------------------------------------
# Define as variáveis independentes (X) e a variável dependente (y)
X = df[['HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H', 'Torque']]
y = df['Cars Prices']

# Divide os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria e treina o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avalia o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coeficientes do modelo: {model.coef_}")
print(f"Intercepto do modelo: {model.intercept_}")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R-quadrado): {r2:.2f}")

# Exibe as previsões vs. os valores reais
results_df = pd.DataFrame({'Preço Real': y_test, 'Previsão do Modelo': y_pred})
print("\nResultados da Previsão:")
print(results_df.head())


# In[ ]:




