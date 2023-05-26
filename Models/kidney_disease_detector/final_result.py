from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import normalize

from Models.kidney_disease_detector.models.RandomForest import RandomForestModel

# # ==============================================
# #             Low Variance Filter
# # ==============================================
#
# data = pd.read_csv('database/kidney_disease_formatted.csv')
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

# data = pd.read_csv('database/kidney_disease_low_variance_filter.csv')
data = pd.read_csv('Models/kidney_disease_detector/database/kidney_disease_low_variance_filter.csv')
X = data.drop(['classification'], axis = 1)
y = data['classification']
# Separate the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

random_forest_model = RandomForestModel()
random_forest_model.fit_the_model(X_train, y_train)

class ProvideResult():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ProvideResult, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_final_result(input_dict: dict) -> str:
        sickness_level = random_forest_model.predict_value(input_dict)
        return sickness_level


# TODO: these need to be deleted from here as it is a test
# provide_result = ProvideResult()
# X_test_values = X_test.head(1)
# print(f"X_test_values:\n {X_test_values}")
# print(f"X_test_values type: {type(X_test_values)}")
# # print(f"X_test SHAPE: {X_test.shape}")
# # print(f"X_test type: {type(X_test)}")
# # print(f"X_test: {X_test}")
# get_final_result = provide_result.get_final_result(X_test_values)
# print(f"Final result: {get_final_result}")