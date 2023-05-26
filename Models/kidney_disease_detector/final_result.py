from sklearn.model_selection import train_test_split
import pandas as pd
from Models.kidney_disease_detector.models.RandomForest import RandomForestModel

# data = pd.read_csv('database/kidney_disease_low_variance_filter.csv')
data = pd.read_csv('../Models/kidney_disease_detector/database/kidney_disease_low_variance_filter.csv')
X = data.drop(['classification'], axis = 1)
y = data['classification']
# Separate the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

random_forest_model = RandomForestModel()
random_forest_model.fit_the_model(X_train, y_train)
random_forest_model.predict_value_and_return_accuracy(X_test, y_test)

class KidneyProvideResult():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(KidneyProvideResult, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_final_result(input_dict: dict) -> str:
        sickness_level = random_forest_model.predict_value(input_dict)
        return sickness_level
