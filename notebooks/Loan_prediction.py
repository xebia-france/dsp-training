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

from src.constants import files, columns as c
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.metrics import f1_score
# -

# # Load data

loans_df = pd.read_csv(os.path.join(files.RAW_DATA, files.LOANS))

train_df, test_df = train_test_split(loans_df, test_size=0.2, random_state=1)

# # Preprocessing

train_df.describe()

train_df.shape

# TODO veut-on suivre cette métrique ?
def null_values_stats(input_df):
    total = input_df.isnull().sum().sort_values(ascending=False)
    percent = (input_df.isnull().sum()/input_df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))


null_values_stats(train_df)

# ## Fill missing values

num_features = c.Loans.num_features()
cat_features = c.Loans.cat_features()

cat_features

# +
pipeline = ColumnTransformer([
    ("num_pipeline", Pipeline(
                [("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler())]), num_features),
    ("cat_pipeline", Pipeline(
                [("imputer", SimpleImputer(strategy="most_frequent")),
                 ("one_hot_encoder", OneHotEncoder(drop="if_binary"))]), cat_features)
]
)

preprocessed_train = pipeline.fit_transform(train_df)
# -

raw_one_hot_cols = pipeline.named_transformers_["cat_pipeline"].named_steps["one_hot_encoder"].get_feature_names()
one_hot_cols = []
for i in range(len(raw_one_hot_cols)):
    one_hot_col_name = cat_features[int(raw_one_hot_cols[i][1])] + raw_one_hot_cols[i][2:]
    one_hot_cols.append(one_hot_col_name)

preprocessed_train_df = pd.DataFrame(preprocessed_train, columns=num_features + one_hot_cols)

preprocessed_train_df.head()

# # Exploration

sns.countplot(y="Gender", hue="Loan_Status", data= loans_df)

sns.heatmap(loans_df.corr())

# # Modeling

y_train = train_df['Loan_Status']
X_train = preprocessed_train
y_test = test_df["Loan_Status"]
X_test = pipeline.transform(test_df)

model = LogisticRegression()
model.fit(X_train, y_train)

# # Prediction

ypred = model.predict(X_test)

# # Evaluation

evaluation = f1_score(y_test, ypred, pos_label="Y")
evaluation
