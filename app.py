from flask import Flask, render_template, request

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model_selection import model_builder
import os
import joblib



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])  # To render Homepage
def home_page():
    return render_template('heartattack_index.html')


@app.route('/math', methods=['POST'])  # This will be called from UI
def math_operation():
    if (request.method == 'POST'):
        variable1 = int(request.form['num1'])
        variable2 = int(request.form['num2'])
        variable3 = int(request.form['num3'])
        variable4 = int(request.form['num4'])
        variable5 = int(request.form['num5'])
        variable6 = int(request.form['num6'])
        variable7 = int(request.form['num7'])
        variable8 = int(request.form['num8'])
        variable9 = int(request.form['num9'])
        variable10 = int(request.form['num10'])
        variable11 = int(request.form['num11'])
        variable12 = int(request.form['num12'])
        variable13 = int(request.form['num13'])

    L =[]
    L.append(int(variable1))
    L.append(int(variable6))
    L.append(int(variable7))
    L.append(int(variable2))
    L.append(int(variable3))   
    L.append(int(variable8))
    L.append(int(variable9))
    L.append(int(variable4))
    L.append(int(variable10))
    L.append(int(variable5))
    L.append(int(variable11))
    L.append(int(variable12))
    L.append(int(variable13))

    df = pd.read_csv("heart.csv")
    df.drop(["target"],axis=1,inplace=True)

    column = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']

    categorical_val = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    data = pd.DataFrame(np.array(L).reshape(1,13), columns = column)

    df = pd.concat([df, data], ignore_index = True)

    dataset = pd.get_dummies(df, columns = categorical_val)

    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.array(dataset.tail(1)))

    if os.path.exists("E:/PROJECTS/HEART-ATTACK-PREDICTION/model.pkl"):
       model = joblib.load(r'E:/PROJECTS/HEART-ATTACK-PREDICTION/model.pkl')
       prediction = model.predict(scaled)[0]
    else:
        x = model_builder()
        model = joblib.load(r'E:/PROJECTS/HEART-ATTACK-PREDICTION/model.pkl')
        prediction = model.predict(scaled)[0]

    if prediction == 1:
        results = "YOUR ARE AT VERY HIGH RISK"
    else:
        results = "YOU ARE HEALTHY"

    return render_template('results.html', result=results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)