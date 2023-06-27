from flask import Flask, render_template, request, Response, jsonify
from Models.kidney_disease_detector.final_result import KidneyProvideResult
from Models.diabetes_detector.final_result import DiabetesProvideResult
from Models.heart_disease_detector.fuzzy_system.final_result import HeartProvideResult

import pandas as pd

app = Flask(__name__)

@app.route('/start',methods=['GET', 'POST'])
def main_page():
    print("Called...")

    response ={
        'result': 'Test OK'
    }
    return jsonify(response)

@app.route('/diabetes_disease/result', methods=['GET', 'POST'])
def diabetes_disease_final_result():
    input_dict = request.json
    print(f"\nDIABETES DISEASE: Got the following values: {input_dict}")

    data_values = list(input_dict.values())
    data = []
    data.append(data_values)
    data_df = pd.DataFrame(data=data, columns=list(input_dict.keys()))

    # need to rearrange columns based on the order they were fit in the model
    column_names = ['blood_pressure','cholestrol','bmi','smoker','physical_activity','fruits',
                    'general_health','mental_health','physical_health','sex','age',
                    'education','income']
    data_df = data_df.reindex(columns=column_names)

    provide_result = DiabetesProvideResult()
    output = provide_result.get_final_result(input_dict=data_df)
    print(f"\nDIABETES DISEASE: Result is: {output}")

    response = {'response': output}
    return jsonify(response)

@app.route('/heart_disease/result', methods=['GET', 'POST'])
def heart_disease_final_result():
    input_dict = request.json
    print(f"\nHEAERT DISEASE: Got the following values: {input_dict}")

    data_values = list(input_dict.values())
    data = []
    data.append(data_values)
    data_df = pd.DataFrame(data=data, columns=list(input_dict.keys()))

    provide_result = HeartProvideResult()
    output = provide_result.get_final_result(input_dict=data_df)
    print(f"\nHEART DISEASE: Result is: {output}")

    response = {'response': output}
    return jsonify(response)

@app.route('/kidney_disease/result', methods=['GET', 'POST'])
def kidney_disease_final_result():
    input_dict = request.json
    print(f"\nKIDNEY DISEASE: Got the following values: {input_dict}")

    data_values = list(input_dict.values())
    data = []
    data.append(data_values)
    data_df = pd.DataFrame(data=data, columns=list(input_dict.keys()))

    # need to rearrange columns based on the order they were fit in the model
    column_names = ['age','blood_pressure','blood_glucose_random','blood_urea',
                    'serum_creatine','sodium','potassium','hemoglobin','packed_cell_volume',
                    'white_blood_cell_count']

    data_df = data_df.reindex(columns=column_names)

    provide_result = KidneyProvideResult()
    output = provide_result.get_final_result(input_dict=data_df)
    print(f"\nKIDNEY DISEASE: Result is: {output}")

    response = {'response': output}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8448, debug=True)