from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


#Função que verifica se os dados foram equilibrados
def checkingSmote(y_train, y_train_resh):
  # Antes do SMOTE
  print("Distribuição original:")
  print(Counter(y_train))

  # Depois do SMOTE
  print("Distribuição reamostrada:")
  print(Counter(y_train_resh))

def printCroosValidation(rf_cv,svm_cv,logreg_cv):
  print('Mean f1 scores:')
  print('Random Forest mean :',rf_cv.mean())
  print('SVM mean :',svm_cv.mean())
  print('Logistic Regression mean :',logreg_cv.mean())

def printF1_score(rf_f1,svm_f1,logreg_f1):
  print('Mean f1 scores:')
  print('RF mean :',rf_f1)
  print('SVM mean :',svm_f1)
  print('LR mean :',logreg_f1)
  
# Função para imprimir métricas de performance
def print_metrics(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Métricas para {model_name}:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    print("\n")

# Função para plotar a matriz de confusão usando seaborn
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Verdadeira')
    plt.show()