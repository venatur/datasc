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
    result1 = datos[label1,:]
    result2 = datos[label2,:]    
    
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
med_clas1 = np.mean(clas1)

med_clas2 = np.mean(clas2)




cov_clas1 = np.cov(clas1[0],rowvar=False)
cov_clas2 = np.cov(clas2[0],rowvar=False)

#prioris
priori1 = lengs(clas1,C_train)
priori2 = lengs(clas2,C_train)

#saneados
san1 = sanear(cov_clas1)
san2 = sanear(cov_clas2)

#transpuestas w inversas
trasp_ct1 = np.transpose(C_test)
trasp_san1 = np.transpose(san1)
med_trasp_clas1 = np.transpose(med_clas1)
inv_san1 = np.linalg.inv(san1)

#logs

log_disc1 = np.log(np.linalg.det(san1))
log_disc2 = np.log(np.linalg.det(san2))
log_priori1 = np.log(priori1)
log_priori2 = np.log(priori2)

c = -0.5
QLD = []
for i in range(len(C_test[0])):
    
    X = C_test[i,:]
    X_T = np.transpose(X)
    sum1 = np.dot(c*X,san1)*X_T
    sub2 = np.dot(med_clas1,inv_san1)
    sum2 = np.dot(sub2,X_T)
    sub3 = np.dot(c*X,med_clas1)*X_T
    sum3 = np.dot(sub3,med_trasp_clas1)
    sum4 = -((c)*log_disc1+log_priori1)
    form = sum1+sum2+sum3+sum4
    QLD.append(form1)
   


print(j)
#plt.scatter(C_train,cl_train)
#plt.tittle('datos de funciones')
#plt.xlabel('datos')
#plt.ylabel('clases')
#plt.show()
 
