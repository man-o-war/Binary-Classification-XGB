#!/usr/bin/env python
# coding: utf-8
# Skywalker's Domain


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,log_loss
from prettytable import PrettyTable

print('Printing Model Performance...')

# Load the data
training_data = pd.read_csv('./Data/training_set.csv', index_col=0)

# Splitting the data for Y
X = training_data.drop(['Y'],axis=1)
y = training_data['Y']

# Train Test data splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)

# Feature selection using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier as RF

FS = RF(100, max_depth=None, n_jobs=-1)
FS.fit(X_train,y_train)
feature_importance = FS.feature_importances_

# Ranking Feature Importance Scores
mean_fi = np.mean(feature_importance)
top_fi = len(np.where(feature_importance > mean_fi)[0])
fi = sorted(zip(X.columns,feature_importance),key=lambda x: x[1], reverse=True)

top_features = [x[0] for x in fi[:30]]

# Extracting subset on the basis of top features
X_train_top_FI = X_train[top_features]
X_test_top_FI = X_test[top_features]

# Data Normalisation using Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train_top_FI)

# Transforming
X_train_top_FI = pd.DataFrame(scaler.transform(X_train_top_FI),columns=X_train_top_FI.columns)
X_test_top_FI = pd.DataFrame(scaler.transform(X_test_top_FI),columns=X_test_top_FI.columns)

# Importing Presaved Model
import pickle

classifier = pickle.load(open('./XGBFTW.sav', 'rb'))

# Making Predictions to get scores
y_train_pred = classifier.predict(X_train_top_FI)
y_train_prob = classifier.predict_proba(X_train_top_FI)[:,1]
y_test_pred = classifier.predict(X_test_top_FI)
y_test_prob = classifier.predict_proba(X_test_top_FI)[:,1]

# Using PrettyTable for printing performance of each model
table = PrettyTable()
table.field_names = ["Model","Train Logloss", "Validation Logloss", "Train AUC", "Validation AUC"]
table.add_row(["Xgboost Performance", log_loss(y_train,y_train_prob), log_loss(y_test,y_test_prob),roc_auc_score(y_train, y_train_prob), roc_auc_score(y_test, y_test_prob)])
print(table)