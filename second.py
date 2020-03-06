# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:55:33 2020

@author: iscca
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
car_data = pd.read_csv('C:/Users/iscca/Documents/Python Scripts/dnasoftware_laptop/Datasets/Datasets/car_price.csv')

plt.scatter(car_data['Year'],car_data['Price'])
plt.title("Year vs Price")
plt.xlabel("Year")
plt.ylabel("price")
plt.show()

features = car_data.iloc[:,0:1]
labels = car_data.iloc[:,1]

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=.20, random_state=0)

lin_reg = LinearRegression()
lin_reg.fit(train_features,train_labels)

predictions = lin_reg.predict(test_features)

comparison = pd.DataFrame({'Real':test_labels,'predictions':predictions})

print("MAE:",metrics.mean_absolute_error(test_labels,predictions))
print("MSE:",metrics.mean_squared_error(test_labels,predictions))
print("RMSE:",np.sqrt(metrics.mean_squared_error(test_labels,predictions)))