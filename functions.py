from collections import Counter

#Função que verifica se os dados foram equilibrados
def checkingSmote(y_train, y_train_resh):
  # Antes do SMOTE
  print("Distribuição original:")
  print(Counter(y_train))

  # Depois do SMOTE
  print("Distribuição reamostrada:")
  print(Counter(y_train_resh))