from flask import Flask, render_template, request
from Models.kidney_disease_detector.final_result import KidneyProvideResult
from Models.diabetes_detector.final_result import DiabetesProvideResult
from Models.heart_disease_detector.fuzzy_system.final_result import HeartProvideResult

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/diabetes_disease/result', methods=['GET', 'POST'])
def diabetes_disease_final_result():
    input_dict = request.form.to_dict()
    print(input_dict)
    provide_result = DiabetesProvideResult()
    output = provide_result.get_final_result(input_dict=input_dict)
    return render_template('diabetes_result.html', output=output)

@app.route('/kidney_disease/result', methods=['GET', 'POST'])
def kidney_disease_final_result():
    input_dict = request.form.to_dict()
    print(input_dict)
    provide_result = KidneyProvideResult()
    output = provide_result.get_final_result(input_dict=input_dict)
    return render_template('kidney_result.html', output=output)

@app.route('/heart_disease/result', methods=['GET', 'POST'])
def heart_disease_final_result():
    input_dict = request.form.to_dict()
    print(input_dict)
    provide_result = HeartProvideResult()
    output = provide_result.get_final_result(input_dict=input_dict)
    return render_template('heart_result.html', output=output)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8448, debug=True)