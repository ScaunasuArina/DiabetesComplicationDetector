from sklearn.model_selection import train_test_split
import pandas as pd
from Models.diabetes_detector.models.SVM import SVMModel

# data = pd.read_csv('database/diabetes_disease_low_variance_filter.csv')
data = pd.read_csv('../Models/diabetes_detector/database/diabetes_disease_low_variance_filter.csv')
X = data.drop(['classification'], axis = 1)
y = data['classification']
# Separate the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

# create the SVM object
svm_model = SVMModel()
svm_model.fit_the_model(X_train, y_train)
svm_model.predict_value_and_return_accuracy(X_test, y_test)

class DiabetesProvideResult():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DiabetesProvideResult, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_final_result(input_dict: dict) -> str:
        sickness_level = svm_model.predict_value(input_dict)
        return sickness_level
