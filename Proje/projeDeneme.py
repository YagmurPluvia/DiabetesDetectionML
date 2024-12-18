# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('diabetes.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,1:8].values #bağımsız değişkenler
y = veriler.iloc[:,8:].values#bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

import statsmodels.api as sm  
print('Forest OLS')
model = sm.OLS(rfc.predict(x),x)
print(model.fit().summary())


print('Forest OLS')
model = sm.OLS(y_pred,y_test)
print(model.fit().summary())

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print('RandomForest')
print(cm)












