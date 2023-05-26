from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

dataset=pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')

dataset = dataset.dropna()
dataset.reset_index(inplace=True)
dataset['Dependents'] = dataset['Dependents'].replace(to_replace = '3+', value = 4)
dataset.replace({'Married':{'Yes':1,'No':0}, 
                 'Gender':{'Male':1,'Female':0},
                 'Education':{'Graduate':1,'Not Graduate':0},
                 'Self_Employed':{'Yes':1,'No':0},
                 'Property_Area':{'Rural':0,'Urban':1,'Semiurban':2}},
                 inplace = True)
dataset['Dependents'] = dataset['Dependents'].astype('int')
X = dataset.iloc[:,2:-1].values
dataset.replace({'Loan_Status':{'Y':1,'N':0}}, inplace = True)
y = dataset.iloc[:,-1].values
x_train, x_test, y_train, y_test =train_test_split(X,y,test_size=0.25,random_state=42)

log_classifier = LogisticRegression()
log_classifier.fit(x_train, y_train)


joblib.dump(log_classifier, 'loan_model.pkl' )




