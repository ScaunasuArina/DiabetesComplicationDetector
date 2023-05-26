from Models.heart_disease_detector.fuzzy_system.fuzzification import Fuzzify
from Models.heart_disease_detector.fuzzy_system.inference import FuzzyInference
from Models.heart_disease_detector.fuzzy_system.defuzzification import Defuzzify

fuzzification_res = Fuzzify()
inference_res = FuzzyInference()
defuzzification_res = Defuzzify()


class ProvideResult(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ProvideResult, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_final_result(input_dict: dict) -> str:
        age, bp, bs, cholesterol, hr, ecg, op, cp, exercise, thallium, sex =\
            fuzzification_res.fuzzification_result(input_dict)

        fuzzy_sickness = \
            inference_res.inference_result(age, bp, bs, cholesterol, hr, ecg, op, cp, exercise, thallium, sex)

        sickness_level = defuzzification_res.defuzzification_result(fuzzy_sickness)

        if 'healthy' in sickness_level:
            sickness_level = 'Absence HD'
        else:
            sickness_level = 'Presence HD'

        return sickness_level

# TODO: this is just an example here to see how this works!!!

# final_res = result.get_final_result({'age':52,
#                          'blood_pressure':242,
#                          'blood_sugar': 166,
#                          'cholestrol': 89,
#                          'heart_rate': 24,
#                          'ecg': 1,
#                          'old_peak': 1.5,
#                          'chest_pain': 3,
#                          'exercise': 0,
#                          'thallium_scan': 6,
#                          'sex': 1})
#
# print(final_res)

import pandas as pd
heart_result  = ProvideResult()
heart_data = pd.read_csv('../database/heart_disease_formatted.csv')
X_heart = heart_data.drop(['slope_of_st', 'no_vessels_fluro', 'classification'], axis = 1)
y_heart = heart_data['classification']
X_test_heart = X_heart.head(1)
y_test_heart = y_heart.head(1)

print(f"y_test_heart:{y_test_heart}")

heart_final_result = heart_result.get_final_result(X_test_heart)
print(f"HEART DISEASE:\n predicted: {heart_final_result}\n real: {y_test_heart}")

