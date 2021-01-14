# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python (dsp-training)
#     language: python
#     name: dsp-training
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

from src.constants import files, columns as c
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
# -

# # Load data

loans_df = pd.read_csv(os.path.join(files.RAW_DATA, files.LOANS_CSV))

train_df, test_df = train_test_split(loans_df, test_size=0.2, random_state=1)

# # Preprocessing

train_df.describe()

train_df.shape


def null_values_stats(input_df):
    total = input_df.isnull().sum().sort_values(ascending=False)
    percent = (input_df.isnull().sum()/input_df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))


null_values_stats(train_df)

# ## Fill missing values

num_variables = c.Loans.num_features()
cat_variables = c.Loans.cat_features()

# +
pipeline = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="most_frequent"), StandardScaler()), num_variables),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder()), cat_variables)
)

# TODO: utiliser https://jorisvandenbossche.github.io/blog/2018/05/28/scikit-learn-columntransformer/

preprocessed_train = pipeline.fit_transform(train_df)
# -

# TODO: trouver un moyen d'accéder aux features
type(pipeline)

preprocessed_train

preprocessed_train_df = pd.DataFrame(preprocessed_train)

preprocessed_train_df.head()

pipeline.named_steps["one_hot_encoding"].get_feature_names()

null_values_stats(preprocessed_train_df)

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


