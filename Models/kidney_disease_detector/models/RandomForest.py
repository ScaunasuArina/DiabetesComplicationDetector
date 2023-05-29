from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# ==============================================
#             Random Forest Model
#                with classes
# ==============================================
# class RandomForestModel(object):
#     def __int__(self):
#         pass
#
#     def fit_the_model(self, X_train, y_train):
#         # self.radomForest_model = RandomForestClassifier(n_estimators = 10)
#         # self.radomForest_model = RandomForestClassifier(n_estimators = 20)
#         self.radomForest_model = RandomForestClassifier()
#         print("\nFitting the model...")
#         start_time = time.time()
#         self.radomForest_model.fit(X_train, y_train)
#         stop_time = time.time()
#
#         print(f"Start time: {start_time}")
#         print(f"Stop time: {stop_time}")
#         print(f"Training duration: {stop_time - start_time} seconds.\n")
#
#     def predict_value_and_return_accuracy(self, X_test, y_test):
#         model_predict = self.radomForest_model.predict(X_test)
#
#         print(f"CONFUSION MATRIX: {confusion_matrix(y_test, model_predict)}\n")
#         print(f"Accuracy is {round(accuracy_score(y_test, model_predict) * 100, 2)}%\n")
#
#     def predict_value(self, X_test):
#         model_predict = self.radomForest_model.predict(X_test)
#         return model_predict


data = pd.read_csv('../database/kidney_disease_low_variance_filter.csv')
# data = pd.read_csv('../Models/kidney_disease_detector/database/kidney_disease_low_variance_filter.csv')
X = data.drop(['classification'], axis = 1)
y = data['classification']
# Separate the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

random_forest_model = RandomForestClassifier()
print("\nFitting the model...")
start_time = time.time()
random_forest_model.fit(X_train, y_train)
stop_time = time.time()

print(f"Start time: {start_time}")
print(f"Stop time: {stop_time}")
print(f"Training duration: {stop_time - start_time} seconds.\n")

# testing the model
model_predict = random_forest_model.predict(X_test)
print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, model_predict)}\n")
print(f"Accuracy is {round(accuracy_score(y_test, model_predict) * 100, 2)}%\n")

# save trained model
pickle.dump(random_forest_model, open('random_forest_model.pkl', 'wb'))
