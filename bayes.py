# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import scipy.io
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
mat = scipy.io.loadmat('datos_wdbc.mat')

#Model gaussiano
model = GaussianNB()
model_d = MultinomialNB()
dic = dict()
def data_positive(data):
    lista = list()
    for i in range(len(data[0])):
        for j in range(len(data)):
            if data[j][i]<=0:
                lista.append(data[j][i])
                data[j][i] = 0.1
    return X_train

def class_separator(data):
    sep = dict()
    for i in range(len(data)):
        vector = data[i]
        class_value = vector[-1]
        if (class_value not in sep):
            sep[class_value] = list()
        sep[class_value].append(vector)
    return sep    

def mean(units):
    return sum(units)/float(len(numbers))

def stdev(units):
    avg = mean(units)
    variance = sum([(x-avg)**2 for x in units]) / float(len(units))
    return np.sqrt(variance)

def calcs(data):
    
    op = [(mean(i), stdev(i), len(i)) for i in data]
    del(op[-1])
    return op

def byclas(data):
    sep = class_separator(data)
    op = {}
    for clas_value, ita in sep.items():
        op[clas_value] = calcs(ita)
    return op

def toCox(data):
    nuevo = []
    
    return data

trn = mat['trn']
clas = trn['y'][0,0]
xc = trn['xc'][0,0]
xd = trn['xd'][0,0]

discrets = pd.DataFrame(data=xd)
discrets

print("DATOS DISCRETOS")
D_train, D_test, c_train, c_test = train_test_split(discrets, clas, random_state=1)
print('Number of rows in the total set: {}'.format(discrets.shape[0]))
print('Number of rows in the training set: {}'.format(D_train.shape[0]))
print('Number of rows in the test set: {}\n'.format(D_test.shape[0]))

model_d.fit(D_train, c_train.ravel())
predict_train = model_d.predict(D_test)
print('Accuracy score: ', format(accuracy_score(c_test, predict_train)))
print('Precision score: ', format(precision_score(c_test, predict_train)))
print('Recall score: ', format(recall_score(c_test, predict_train)))
print('F1 score: ', format(f1_score(c_test, predict_train))+'\n')


print("DATOS CONTINUOS")
X_train, X_test = train_test_split(xc,test_size=0.2)
lenx = len(X_train)
clasi = clas[:lenx]
lista = data_positive(X_train)


#Box Cox
#cosa = data_positive(X_train)
#train_data, fitted_lambda = stats.boxcox(lista.all())
#test_data = stats.boxcox(X_test, fitted_lambda)


# (optional) plot train & test
#fig, ax=plt.subplots(1,2)
#sns.distplot(train_data, ax=ax[0])
#sns.distplot(test_data, ax=ax[1])

#Train model
model.fit(lista,clasi.ravel())

predict_train = model.predict(lista)

mat = confusion_matrix(predict_train, clasi)
names = np.unique(predict_train)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')

glued = np.append(xc,clas,axis=1)
separated = class_separator(X_test)
print('Accuracy score: ', format(accuracy_score(clasi, predict_train)))
print('Precision score: ', format(precision_score(clasi, predict_train)))
print('Recall score: ', format(recall_score(clasi, predict_train)))
print('F1 score: ', format(f1_score(clasi, predict_train)))