from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# ==============================================
#               SVM Model
#             with Classes
# ==============================================
# class SVMModel:
#     def __int__(self):
#         pass
#
#     def fit_the_model(self, X_train, y_train):
#         # svm_model = svm.SVC(kernel='rbf')
#         self.svm_model = svm.SVC(kernel='linear')
#         print("\nFitting the model...")
#
#         start_time = time.time()
#         self.svm_model.fit(X_train, y_train)
#         stop_time = time.time()
#
#         print(f"Start time: {start_time}\n")
#         print(f"Stop time: {stop_time}\n")
#         print(f"Training duration: {stop_time - start_time} seconds.")
#
#     def predict_value_and_return_accuracy(self, X_test, y_test):
#         model_predict = self.svm_model.predict(X_test)
#
#         print(f"CONFUSION MATRIX: {confusion_matrix(y_test, model_predict)}\n")
#         print(f"Accuracy is {round(accuracy_score(y_test, model_predict) * 100, 2)}%\n")
#
#     def predict_value(self, X_test):
#         model_predict = self.svm_model.predict(X_test)
#         return model_predict


data = pd.read_csv('../database/diabetes_disease_low_variance_filter.csv')
# data = pd.read_csv('../Models/diabetes_detector/database/diabetes_disease_low_variance_filter.csv')
X = data.drop(['classification'], axis = 1)
y = data['classification']
# Separate the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

# create the SVM object
svm_model = svm.SVC(kernel='linear')
print("\nFitting the model...")
start_time = time.time()
svm_model.fit(X_train, y_train)
stop_time = time.time()

print(f"Start time: {start_time}\n")
print(f"Stop time: {stop_time}\n")
print(f"Training duration: {stop_time - start_time} seconds.")

# testing the model
model_predict = svm_model.predict(X_test)
print(f"CONFUSION MATRIX: {confusion_matrix(y_test, model_predict)}\n")
print(f"Accuracy is {round(accuracy_score(y_test, model_predict) * 100, 2)}%\n")

# save trained model
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))