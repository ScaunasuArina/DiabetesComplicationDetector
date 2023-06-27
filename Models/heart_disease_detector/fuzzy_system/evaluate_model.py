from Models.heart_disease_detector.fuzzy_system.fuzzification import Fuzzify
from Models.heart_disease_detector.fuzzy_system.inference import FuzzyInference
from Models.heart_disease_detector.fuzzy_system.defuzzification import Defuzzify

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np

fuzzification_res = Fuzzify()
inference_res = FuzzyInference()
defuzzification_res = Defuzzify()

# load database
data = pd.read_csv('../database/heart_disease_formatted.csv')
X = data.drop(['slope_of_st', 'no_vessels_fluro', 'classification'], axis = 1)
y = data['classification']

model_predict = []
for entry in range(X.shape[0]):
    # for each entry in database, create an input dict
    df_entry = X.iloc[entry]
    input_dict = {}
    keys_list = ['age','sex','chest_pain','blood_pressure','cholestrol','blood_sugar','ecg','heart_rate',
                 'exercise','old_peak', 'thallium_scan']
    for k in keys_list:
        input_dict[k]=None

    # get data from database
    for k in keys_list:
        input_dict[k] = df_entry[k]

    # for each entry in database, perform prediction and store these values in list model_predict
    age, bp, bs, cholesterol, hr, ecg, op, cp, exercise, thallium, sex = \
        fuzzification_res.fuzzification_result(input_dict)

    fuzzy_sickness = \
        inference_res.inference_result(age, bp, bs, cholesterol, hr, ecg, op, cp, exercise, thallium, sex)

    sickness_level = defuzzification_res.defuzzification_result(fuzzy_sickness)
    if 'healthy' in sickness_level:
        sickness_level = 0  # Absence of heart disease
    else:
        sickness_level = 1  # Presence of heart disease

    # save the predicted value to model_predict
    model_predict.append(sickness_level)

model_predict = np.array(model_predict)

# Calculate performance metrics
print(f"Accuracy is {round(accuracy_score(y, model_predict) * 100, 2)}%")
print(f"MSE is {mean_squared_error(y, model_predict)}")
print(f"MAE is {mean_absolute_error(y, model_predict)}")
print(f"F1-score is {round(f1_score(y, model_predict) * 100, 2)}%")
print(f"Precission is {round(precision_score(y, model_predict) * 100, 2)}%")
print(f"Recall is {round(recall_score(y, model_predict) * 100, 2)}%")

print((2* 89.56* 15.38)/(89.56+15.38))