#==============  Importando bibliotecas importantes ==================#
import numpy as np # Faz os calculos de algebra linear
import pandas as pd # Para a leitura de dados

#importando bibliotecas do pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Configurando a opção do pandas para evitar avisos de downcasting
pd.set_option('future.no_silent_downcasting', True)

#==============  Iniciando código ==================#

#Leitura do arquivo (database) e exibindo as 3 primeiras linhas
df = pd.read_csv('database/healthcare-dataset-stroke-data.csv')
df.head(3)
#print(df.head(3))

#Verificando dados nulos e somando em seguida para saber quais colunas tem dados nulos
df.isnull().sum()
#print(df.isnull().sum())

#Usando pipeline para execução de multiplas instruções 
#Arvore de decisão foi escolhida
DT_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=42))
                              ])

#Fazendo uma copia dos DF na seguintes colunas que foram declaradas importantes
X = df[['age','gender','bmi']].copy()
#Númerando o tipos de sexos
X.gender = X.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)

#Retorna os valores que são NaN dentro do dataframe X
Missing = X[X.bmi.isna()]
#Retorna os valores que não são NaN dentro do dataframe X
X = X[~X.bmi.isna()]

#Removendo a coluna "bmi"
Y = X.pop('bmi')
print(Y)