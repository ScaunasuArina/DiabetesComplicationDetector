import pickle

# load pre-trained model
# random_forest_model = pickle.load(open('models/svm_model.pkl', 'rb'))
svm_model = pickle.load(open('../Models/diabetes_detector/models/svm_model.pkl', 'rb'))

class DiabetesProvideResult():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DiabetesProvideResult, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_final_result(input_dict: dict) -> str:
        sickness_dict = {
            '0': 'PREZENT',
            '1': 'ABSENT'
        }
        sickness_level = svm_model.predict(input_dict)
        return str(sickness_dict[str(sickness_level[0])])
