# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 04:54:59 2018

@author: muralish
"""

import pandas as pd
train = pd.read_csv('bf_train_cleaned.csv')
test = pd.read_csv('bf_test_cleaned.csv')
train_head = train.head(5)


x_train = train.iloc[:,:-1]
y_train = train.iloc[:,11:12]
x_test = test
#Label Encoder
from sklearn.preprocessing import LabelEncoder
categories = ['Gender','Age','Occupation','City_Category',
              'Stay_In_Current_City_Years','Marital_Status',
              ]
le = LabelEncoder()
for i in categories:
    x_train[i] = le.fit_transform(x_train[i])
x_train = pd.get_dummies(x_train, columns=categories)

for i in categories:
    x_test[i] = le.fit_transform(x_test[i])
x_test = pd.get_dummies(x_test, columns=categories)


for cols in categories:
     column = cols + '_0'
     x_train= x_train.drop(column,axis=1)
     
for cols in categories:
     column = cols + '_0'
     x_test= x_test.drop(column,axis=1)




#Regression
"""from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn import ensemble, metrics
from sklearn import linear_model, preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
import numpy as np

x_train = x_train.iloc[:,2:]
x_test = x_test.iloc[:,2:]

lasso = Lasso(alpha=0.001, max_iter=10000, random_state=42)
lasso.fit(x_train, np.log1p(y_train))

KRR = KernelRidge(alpha=0.05, kernel='polynomial', degree=1, coef0=2.5)
lasso = linear_model.Lasso(alpha=0.0005, max_iter=10000, random_state=42)
GBoost = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, 
                                            max_features='sqrt', loss='huber', random_state=42)

reg = AveragingModels(models=(KRR, lasso, GBoost), weight=[0.30, 0.30, 0.40])

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weight):
        self.models = models
        self.weight = weight
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([(model.predict(X) * weight) for model, weight in zip(self.models_, self.weight)])
        return np.sum(predictions, axis=1)

def rmse_cv(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

score = rmse_cv(reg, x_train, np.log1p(y_train))
print(round(score.mean(), 5))"""










#Additional
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth = 7,splitter= 'best')
regressor.fit(x_train,y_train)
prediction = regressor.predict(x_test)

submission = pd.DataFrame({
    "Prediction": prediction
})
submission.to_csv("submission.csv", index=False)