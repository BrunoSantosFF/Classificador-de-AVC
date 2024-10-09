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

#Y está recebendo a coluna "bmi"
Y = X.pop('bmi')

#Fazendo o treinamento (para aprender a relação entre age, gender e bmi.)
DT_bmi_pipe.fit(X,Y)

#Fazendo a predição dos valores de bmi(NaN) com base nas colunas gender e age usando o treinamento
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age', 'gender']]), index=Missing.index)
#print(predicted_bmi)

#Os valores ausentes de bmi no DataFrame original (df) são preenchidos com os valores preditos pelo modelo.
df.loc[Missing.index, 'bmi'] = predicted_bmi

#Confirmando se o processo deu certo ou se ainda há algum valor nulo
#print('Missing values: ',sum(df.isnull().sum()))
