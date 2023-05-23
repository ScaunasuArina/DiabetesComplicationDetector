from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import time

import pandas as pd
from sklearn.preprocessing import normalize

# TODO: might delete the commented code from here
# ==============================================
#             Low Variance Filter
# ==============================================
#
# data = pd.read_csv('../database/kidney_disease_formatted.csv')
# print(f"SHAPE:\n{data.shape}\n\n")
# print(f"HEAD:\n{data.head()}\n\n")
# print(f"\nINFO:\n{data.info()}\n\n")
# print(f"DESCRIPTION:\n{data.describe(include='all')}\n\n")
#
# # sepparating class attribute from input attributes for classification algorithms
# X = data.drop(['classification'], axis = 1)
# y = data['classification']
#
# print(f"X.columns: {X.columns}\n\n")
#
# # before we calculate the variance of each variable, we need to make normalization
# normalize = normalize(X)
# X_scaled = pd.DataFrame(normalize)
# print(f"X_scaled.var:{X_scaled.var()}\n")
# print(f"X_scaled SHAPE:\n{X_scaled.shape}\n\n")
#
# # storing the variance and name of variables
# variance = X_scaled.var()
# columns = X.columns
#
# # saving the names of variables having variance more than a threshold value
# variable = [ ]
#
# for i in range(0,len(variance)):
#     # if variance[i]>=0.000000003: # we keep 17/25 attributes with this threshold
#     # if variance[i]>=0.000000004: # we keep 14/25 attributes with this threshold
#     # if variance[i]>=0.00000001: # we keep 13/25 attributes with this threshold
#     if variance[i]>=0.0000001: # we keep 10/25 attributes with this threshold
#         variable.append(columns[i])
#
# print(f"Variables list legth: {len(variable)}\n")
# print(f"Variables: {variable}\n\n")
#
# # creating a new dataframe using the above variables
# X = X[variable]
# print(f"New X SHAPE: {X.shape}\n\n")
#
# # Separate the data in train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

# ==============================================
#             Random Forest Model
#               without classes
# ==============================================
#
# # model = RandomForestClassifier(n_estimators = 10)
# # model = RandomForestClassifier(n_estimators = 20)
# radomForest_model = RandomForestClassifier()
#
# print("\nFitting the model...")
# start_time = time.time()
# radomForest_model.fit(X_train, y_train)
# stop_time = time.time()
#
# print(f"Start time: {start_time}\n")
# print(f"Stop time: {stop_time}\n")
# print(f"Training duration: {stop_time - start_time} seconds.")
#
# # Try a prediction
# model_predict = radomForest_model.predict(X_test)
#
# print(f"CONFUSION MATRIX: {confusion_matrix(y_test, model_predict)}\n")
# print(f"Accuracy is {round(accuracy_score(y_test, model_predict)*100, 2)}%\n")

# ==============================================
#             Random Forest Model
#                with classes
# ==============================================
class RandomForestModel(object):
    def __int__(self):
        pass

    def fit_the_model(self, X_train, y_train):
        # self.radomForest_model = RandomForestClassifier(n_estimators = 10)
        # self.radomForest_model = RandomForestClassifier(n_estimators = 20)
        self.radomForest_model = RandomForestClassifier()
        print("\nFitting the model...")
        start_time = time.time()
        self.radomForest_model.fit(X_train, y_train)
        stop_time = time.time()

        print(f"Start time: {start_time}\n")
        print(f"Stop time: {stop_time}\n")
        print(f"Training duration: {stop_time - start_time} seconds.")

    def predict_value_and_return_accuracy(self, X_test, y_test):
        model_predict = self.radomForest_model.predict(X_test)

        print(f"CONFUSION MATRIX: {confusion_matrix(y_test, model_predict)}\n")
        print(f"Accuracy is {round(accuracy_score(y_test, model_predict) * 100, 2)}%\n")

    def predict_value(self, X_test):
        model_predict = self.radomForest_model.predict(X_test)
        return model_predict

# create the object
# random_forest_model = RandomForestModel()
# random_forest_model.fit_the_model(X_train, y_train)
# random_forest_model.predict_value(X_test, y_test)

