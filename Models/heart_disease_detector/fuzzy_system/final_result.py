from Models.heart_disease_detector.fuzzy_system.fuzzification import Fuzzify
from Models.heart_disease_detector.fuzzy_system.inference import FuzzyInference
from Models.heart_disease_detector.fuzzy_system.defuzzification import Defuzzify

fuzzification_res = Fuzzify()
inference_res = FuzzyInference()
defuzzification_res = Defuzzify()


class HeartProvideResult(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(HeartProvideResult, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_final_result(input_dict: dict) -> str:
        age, bp, bs, cholesterol, hr, ecg, op, cp, exercise, thallium, sex =\
            fuzzification_res.fuzzification_result(input_dict)

        fuzzy_sickness = \
            inference_res.inference_result(age, bp, bs, cholesterol, hr, ecg, op, cp, exercise, thallium, sex)

        sickness_level = defuzzification_res.defuzzification_result(fuzzy_sickness)

        if 'healthy' in sickness_level:
            sickness_level = 'ABSENT'  # Absence of heart disease
        else:
            sickness_level = 'PREZENT'  # Presence of heart disease

        return sickness_level
