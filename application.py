import pandas as pd
import joblib

model = joblib.load('loan_model.pkl') 

user_input = {
    'Gender': 1,
    'Education':1,
    'Married': 1, 
    'Self_Employed': 1,
    'Property_Area': 0,
    'Dependents' : 0,
    'ApplicantIncome' : 1000,
    'CoapplicantIncome' : 1500,
    'LoanAmount' : 114,
    'Loan_Amount_Term' : 123,
    'Credit_History' : 1
}

user_data = pd.DataFrame(user_input, index=[0])

prediction = model.predict(user_data)

if prediction[0] == 1:
    print("Loan Approval: Yes")
else:
    print("Loan Approval: No")