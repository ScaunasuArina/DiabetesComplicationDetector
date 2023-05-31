from Models.kidney_disease_detector.final_result import KidneyProvideResult
from Models.diabetes_detector.final_result import DiabetesProvideResult
from Models.heart_disease_detector.fuzzy_system.final_result import HeartProvideResult
import pandas as pd

kidney_data = pd.read_csv('../Models/kidney_disease_detector/database/kidney_disease_low_variance_filter.csv')
X_kidney = kidney_data.drop(['classification'], axis = 1)
y_kidney = kidney_data['classification']

diabetes_data = pd.read_csv('../Models/diabetes_detector/database/diabetes_disease_low_variance_filter.csv')
X_diabetes = diabetes_data.drop(['classification'], axis = 1)
y_diabetes = diabetes_data['classification']

heart_data = pd.read_csv('../Models/heart_disease_detector/database/heart_disease_formatted.csv')
X_heart = heart_data.drop(['slope_of_st', 'no_vessels_fluro', 'classification'], axis = 1)
y_heart = heart_data['classification']

X_test_kidney = X_kidney.head(1)
y_test_kidney = list(y_kidney.head(1))

X_test_diabetes = X_diabetes.head(1)
y_test_diabetes = list(y_diabetes.head(1))

X_test_heart = X_heart.head(1)
y_test_heart = list(y_heart.head(1))

# create the models for each class and predict values
kidney_result = KidneyProvideResult()
diabetes_result = DiabetesProvideResult()
heart_result = HeartProvideResult()

kidney_final_result = kidney_result.get_final_result(X_test_kidney)
diabetes_final_result = diabetes_result.get_final_result(X_test_diabetes)
heart_final_result = heart_result.get_final_result(X_test_heart)
print(f"\n\nKIDNEY DISEASE:\n predicted class: {kidney_final_result[0]}\n real class: {y_test_kidney[0]}")
print(f"\n\nDIABETES DISEASE:\n predicted class: {diabetes_final_result[0]}\n real class: {y_test_diabetes[0]}")
print(f"\n\nHEART DISEASE:\n predicted class: {heart_final_result[0]}\n real class: {y_test_heart[0]}")

print(f"\nDIABETES size: {X_test_diabetes.shape}")
print(f"KIDNEY size: {X_test_kidney.shape}")
print(f"HEART size: {X_test_heart.shape}")