# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:59:03 2020

@author: iscca
"""

from naive_bayes import NaiveBayes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
import pandas as pd

def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
    
mat = scipy.io.loadmat('datos_wdbc.mat')
nb = NaiveBayes()
trn = mat['trn']
clas = trn['y'][0,0]
xc = trn['xc'][0,0]
xd = trn['xd'][0,0]

#datos discretos
discrets = pd.DataFrame(data=xd)
#datos continuos
continuos = pd.DataFrame(data=xc)

D_train, D_test, c_train, c_test = train_test_split(discrets, clas, shuffle=False)

x=[1,2,3,4,5,0,0,0,0]
y=[0,0,0,0,0]

for c in range(len(x)):
    xy = x[y==c]
nb.fit(D_train, c_train)
predictions = nb.predict(D_test)
