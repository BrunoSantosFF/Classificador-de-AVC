from collections import Counter

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