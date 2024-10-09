#==============  Importando bibliotecas importantes ==================#
import numpy as np # Faz os calculos de algebra linear
import pandas as pd # Para a leitura de dados

#importando bibliotecas do pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

#importando o arquivo de graficos
from plot_graphics import create_plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#importando para treino e teste
from sklearn.model_selection import train_test_split

#importando SMOTE 
from imblearn.over_sampling import SMOTE

#importando arquivo de funções auxiliares 
from functions import checkingSmote

#importando modelos
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#importando para validação cruzada 
from sklearn.model_selection import cross_val_score

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

#verifica se teve ou nao AVC
str_only = df[df['stroke'] == 1]
no_str_only = df[df['stroke'] == 0]

#plotar os grafícos, ainda há um erro
#create_plots(str_only, no_str_only)

#Passando valores numeros para valores categoricos
df['gender'] = df['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
df['Residence_type'] = df['Residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)
df['work_type'] = df['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)

#Calculando o recall
#print('Recall: ',100* (249/(249+4861)), end=" %\n")


#=========================  Modelando  ======================# 
X  = df[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']]
y = df['stroke']

#Dividindo os dados em treino e test (30% test e 70% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

#Modelando 
#SMOTE = balancear os dados do database
#Já que existe poucos casos com AVC, usamos o smote para equilibrar os dados
oversample = SMOTE()
X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train.to_numpy())

#função para imprimir dados originais e balanceados
#checkingSmote(y_train, y_train_resh)

#========== Modelos ===============#

rf_pipeline = Pipeline(steps = [('scale',StandardScaler()),('RF',RandomForestClassifier(random_state=42))])
svm_pipeline = Pipeline(steps = [('scale',StandardScaler()),('SVM',SVC(random_state=42))])
logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('LR',LogisticRegression(random_state=42))])

#========= Validação cruzada ========# 
#garantir que o modelo generaliza bem em dados não vistos e ajuda a evitar o overfitting (quando o modelo aprende muito bem os dados de treinamento, mas falha em prever novos dados).
rf_cv = cross_val_score(rf_pipeline,X_train_resh,y_train_resh,cv=10,scoring='f1')
svm_cv = cross_val_score(svm_pipeline,X_train_resh,y_train_resh,cv=10,scoring='f1')
logreg_cv = cross_val_score(logreg_pipeline,X_train_resh,y_train_resh,cv=10,scoring='f1')
