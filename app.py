from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
try:
    model = joblib.load(r"Loan_model6.pkl")
    print(f'Model loaded successfully: {type(model)}')  # Check the type of the loaded model
except Exception as e:
    print(f'Error loading model: {e}')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = request.form
    gender = 1 if data['Gender'] == 'Male' else 0
    married = 1 if data['Married'] == 'Yes' else 0
    dependents = int(data['Dependents'])
    education = int(data['Education'])
    self_employed = int(data['Self_Employed'])
    applicant_income = float(data['ApplicantIncome'])
    coapplicant_income = float(data['CoapplicantIncome'])
    loan_amount = float(data['LoanAmount'])
    loan_amount_term = float(data['Loan_Amount_Term'])
    credit_history = int(data['Credit_History'])
    property_area = int(data['Property_Area'])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[gender, married, dependents, education, self_employed,
                                 applicant_income, coapplicant_income, loan_amount,
                                 loan_amount_term, credit_history, property_area]],
                               columns=['Gender', 'Married', 'Dependents', 'Education',
                                        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                                        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    print(input_data)  # Print the input data for debugging

    # Make prediction
    try:
        prediction = model.predict(input_data)
        result = "Congratulations! Your loan application has been approved." if prediction[0] == 1 else "Sorry, your loan application has been denied."
        return jsonify({'result': result, 'loan_amount': loan_amount})
    except Exception as e:
        print(f'Error during prediction: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)