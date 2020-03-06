# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:50:45 2020

@author: iscca
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

patient_data = pd.read_csv('C:/Users/iscca/Documents/Python Scripts/dnasoftware_laptop/Datasets/Datasets/patients.csv')
features = patient_data.iloc[:,0:3].values
labels = patient_data.iloc[:,3].values


imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(features[:,1:2])
features[:,1:2] = imputer.transform(features[:,1:2])

labelencoder_features = LabelEncoder()
features[:,2] = labelencoder_features.fit_transform(features[:,2])
labels = labelencoder_features.fit_transform(labels)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=.25, random_state=0)

feature_scaler = StandardScaler()
train_features = feature_scaler.fit(train_features)
test_features = feature_scaler.fit(test_features)
