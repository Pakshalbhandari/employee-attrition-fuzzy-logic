# app.py
from flask import Flask, render_template, request
from model import predict_attrition_likelihood

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'OverTime_Yes': int(request.form['overtime']),
            'YearsAtCompany': float(request.form['years_at_company']),
            'Age': float(request.form['age']),
            'WorkLifeBalance': float(request.form['work_life_balance']),
            'EnvironmentSatisfaction': float(request.form['environment_satisfaction']),
            'JobInvolvement': float(request.form['job_involvement']),
            'Gender_Male': int(request.form['gender'])
        }
        likelihood = predict_attrition_likelihood(input_data)
        return render_template('index.html', likelihood=likelihood)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
