import pickle

# load pre-trained model
# random_forest_model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
random_forest_model = pickle.load(open('../Models/kidney_disease_detector/models/random_forest_model.pkl', 'rb'))

class KidneyProvideResult():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(KidneyProvideResult, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_final_result(input_dict: dict) -> str:
        sickness_dict = {
            '0': 'ABSENT',
            '1': 'PREZENT'
        }
        sickness_level = random_forest_model.predict(input_dict)
        return str(sickness_dict[str(sickness_level[0])])
