# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:49:51 2020
@author: iscca
"""
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
import pandas as pd
import sklearn as sk
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def medias(data):
        x = []
        for i in range(len(data[0])):
            x.append(np.mean(data[:,i]))
        return x

def sanear(datos):
    D, V = LA.eig(datos)
    datos[datos<0] =.001
    tras = np.transpose(V)
    op = (D*V)
    ops = np.dot(op,tras)
    return ops

def clasificar(datos,clases):
    label1 = np.where(clases==1)
    label2 = np.where(clases==2)
    result1 = datos[label1[0],:]
    result2 = datos[label2[0],:]    
    
    return result1, result2
    
def lengs(data,training):
    priori = len(data[0])/len(training)       
    return priori

mat = scipy.io.loadmat('datos_wdbc.mat')
trn = mat['trn']
clas = trn['y'][0,0]
xc = trn['xc'][0,0]
xd = trn['xd'][0,0]


#datos continuos
continuos = pd.DataFrame(data=xc)

C_train, C_test, cl_train, cl_test = train_test_split(xc, clas, test_size=.20, shuffle=False)
cov_xc = np.cov(C_train,rowvar=False)
mios = sanear(cov_xc)
clas1, clas2 = clasificar(C_train,cl_train)


#medias

med_clas1 = medias(clas1)
med_clas2 = medias(clas2)


cov_clas1 = np.cov(clas1,rowvar=False)
cov_clas2 = np.cov(clas2,rowvar=False)

#prioris
priori1 = lengs(clas1,C_train)
priori2 = lengs(clas2,C_train)

#saneados
san1 = sanear(cov_clas1)
san2 = sanear(cov_clas2)

#transpuestas w inversas
med_trasp_clas1 = np.transpose(med_clas1)
med_trasp_clas2 = np.transpose(med_clas2)
inv_san1 = np.linalg.inv(san1)
inv_san2 = np.linalg.inv(san2)

#logs

log_disc1 = np.log(np.linalg.det(san1))
log_disc2 = np.log(np.linalg.det(san2))
log_priori1 = np.log(priori1)
log_priori2 = np.log(priori2)

c = -0.5

#Quadratic Discriminant Classfier
QLD1 = []
QLD2 = []
for i in range(len(C_test)):
    
    X = C_test[i,:]
    X_T = np.transpose(X)
    sum1 = c*np.dot(X,san1)*X_T
    sum22 = np.dot(c*X,san2)*X_T
    sub2 = np.dot(med_clas1,inv_san1)
    sub22 = np.dot(med_clas2,inv_san2)
    sum2 = np.dot(sub2,X_T)
    sum222 = np.dot(sub22,X_T)
    sub3 = np.dot(c*X,med_clas1)*X_T
    sub33 = np.dot(c*X,med_clas2)*X_T
    sum3 = np.dot(sub3,med_trasp_clas1)
    sum33 = np.dot(sub33,med_trasp_clas2)
    sum4 = -((c)*log_disc1+log_priori1)
    sum44 = -((c)*log_disc1+log_priori2)
    form1 = sum1+sum2+sum3+sum4
    form2 = sum22+sum222+sum33+sum44
    QLD1.append(form1)
    QLD2.append(form2)
    
# Linear Discriminant Classifier
    
    

#plt.scatter(C_train,cl_train)
#plt.tittle('datos de funciones')
#plt.xlabel('datos')
#plt.ylabel('clases')
#plt.show()