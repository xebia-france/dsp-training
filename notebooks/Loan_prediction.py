# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: dsp
#     language: python
#     name: dsp
# ---

# +
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import sys
sys.path.append("..")

from src.constants import files
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# -

# # Load data

loans_df = pd.read_csv(os.path.join(files.RAW_DATA, files.LOANS_CSV))

loans_df.dtypes

# # Preprocessing

loans_df.describe()

loans_df.shape

total = loans_df.isnull().sum().sort_values(ascending=False)
percent = (loans_df.isnull().sum()/loans_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# ## Fill missing values

loans_df['Gender'] = loans_df['Gender'].fillna(loans_df['Gender'].dropna().mode().values[0] )
loans_df['Married'] = loans_df['Married'].fillna(loans_df['Married'].dropna().mode().values[0] )
loans_df['Dependents'] = loans_df['Dependents'].fillna(loans_df['Dependents'].dropna().mode().values[0] )
loans_df['Self_Employed'] = loans_df['Self_Employed'].fillna(loans_df['Self_Employed'].dropna().mode().values[0] )
loans_df['LoanAmount'] = loans_df['LoanAmount'].fillna(loans_df['LoanAmount'].dropna().median() )
loans_df['Loan_Amount_Term'] = loans_df['Loan_Amount_Term'].fillna(loans_df['Loan_Amount_Term'].dropna().mode().values[0] )
loans_df['Credit_History'] = loans_df['Credit_History'].fillna(loans_df['Credit_History'].dropna().mode().values[0] )

# ### TODO Add scikit learn transformer to fill missing values
# TODO ajouter slide jupytext

total = loans_df.isnull().sum().sort_values(ascending=False)
percent = (loans_df.isnull().sum()/loans_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

code_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}
loans_df = loans_df.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)
df_test = df_test.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)
#drop the uniques loan id
loans_df.drop('Loan_ID', axis = 1, inplace = True)

Dependents_ = pd.to_numeric(loans_df.Dependents)
Dependents__ = pd.to_numeric(df_test.Dependents)
loans_df.drop(['Dependents'], axis = 1, inplace = True)
df_test.drop(['Dependents'], axis = 1, inplace = True)
loans_df = pd.concat([loans_df, Dependents_], axis = 1)
df_test = pd.concat([df_test, Dependents__], axis = 1)

# # Exploration

sns.countplot(y="Gender", hue="Loan_Status", data= loans_df)

sns.heatmap(loans_df.corr())

# # Modeling

y = loans_df['Loan_Status']
X = loans_df.drop('Loan_Status', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)

# # Prediction

ypred = model.predict(X_test)

# # Evaluation

evaluation = f1_score(y_test, ypred)
evaluation


