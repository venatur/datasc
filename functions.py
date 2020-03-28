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
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def medias(data):
    x = []
    for i in range(len(data[1])):
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
    priori = data.shape[0]/training.shape[0]       
    return priori

def QDC(data,inversa,median,san,priori):
    c= -.5
    dt = np.transpose(data)
    media_t = np.transpose(median)
    
    s1 = np.dot(data, inversa)
    sum1 = c*np.dot(s1,dt)
    
    s2 = np.dot(median, inversa)
    sum2 = np.dot(s2,dt)
    
    sum3 = c * np.dot(s2, media_t)
    
    sum4 = c * math.log(np.linalg.det(san))
    
    sum5 = math.log(priori)
    
    r = sum1+sum2+sum3+sum4+sum5   
    return r

def LDC(data,median,inversa,priori):
    c= -.5
    dt = np.transpose(data)
    media_t = np.transpose(median)
    s1 = np.dot(median,inversa)
    
    suma1 = np.dot(s1,dt)
    
    suma2 = c * np.dot(s1, media_t)
    
    suma3 = math.log(priori)
    
    r = suma1 + suma2 +suma3
    return r

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

#saneados
san1 = sanear(cov_clas1)
san2 = sanear(cov_clas2)
#prioris
priori1 = lengs(clas1,C_train)
priori2 = lengs(clas2,C_train)

med_trasp_clas1 = np.transpose(med_clas1)
inv_san1 = np.linalg.inv(san1)
inv_san2 = np.linalg.inv(san1)
mios_t = np.linalg.inv(mios)


c = -0.5
QDC1 = []
QDC2 = []
LDC1 = []
LDC2 = []

for i in range(len(C_test)):
    
    Test = C_test[i, :]
    sumaT1 = QDC(Test,inv_san1,med_clas1,san1,priori1)
    QDC1.append(sumaT1)
    sumaT2 = QDC(Test,inv_san2,med_clas2,san2,priori2)
    QDC2.append(sumaT2)
    sumaL1 = LDC(Test,med_clas1,mios_t,priori1)
    LDC1.append(sumaL1)
    sumaL2 = LDC(Test,med_clas2,mios_t,priori2)
    LDC2.append(sumaL2)



QDC_T = []
LDC_T = []    
for j in range(len(QDC1)):
    if QDC1[j]> QDC2[j]:
        QDC_T.append(1)
    else:
        QDC_T.append(2)

    if LDC1[j] > LDC2[j]:
        LDC_T.append(1)
    else:
        LDC_T.append(2)
        
alpha = np.arange(0,1,.1)
matris = np.dot(san1,mios)
QDC_reg1 = alpha.ravel * matris
#QDC_reg2 = alpha * np.dot(san2,mios)

gama = np.arange(0,1,.1)

LDC_reg1 = gama * mios + (1 - gama) *  np.var(C_train)
#plt.scatter(C_train,cl_train)
#plt.tittle('datos de funciones')
#plt.xlabel('datos')
#plt.ylabel('clases')
#plt.show()
 
