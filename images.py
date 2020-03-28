# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:51:36 2020

@author: iscca
"""
import io
from PIL import Image 
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def cambiar(data):
    for item in range(len(data)):
        if item[i]==255:
            item[i] = 1

    return data

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


newl = []
newt = []
clases = [0,0,0,0,1,1,1,1]
for name in glob.glob('circulos_cuadrados/*.jpg'):
    print(name)
    im = Image.open(name)
    thresh = 200
    fn = lambda x : 1 if x > thresh else 0
    im = im.convert('L').point(fn)
        
    tama = np.size(im)
    dimen = tama[0]*tama[1]
    data = np.asarray(im)
    newv = np.matrix.flatten(data,'C')
    newl.append(newv)
    
        
        #im.show()


test = 'circulos_cuadrados/test/*.jpg'

for name in glob.glob(test):
    print(name)
    im = Image.open(name)
    thresh = 200
    fn = lambda x : 1 if x > thresh else 0
    im = im.convert('L').point(fn)
        
    tama = np.size(im)
    dimen = tama[0]*tama[1]
    data = np.asarray(im)
    newv = np.matrix.flatten(data,'C')
  
    newt.append(newv)

    #im.show()

newl = np.asmatrix(newl)
newt = np.asmatrix(newt)

pca = PCA(n_components=8)
pca = pca.fit(newl)
cosa =pca.components_

mult1 = np.dot(cosa,newl.T)
mult2 = np.dot(cosa,newt.T)

clases = np.array([0,0,0,0,1,1,1,1])

clas1 = np.asmatrix(np.where(clases == 0))
clas2 = np.asmatrix(np.where(clases == 1))

c, r = clas1.shape
c2, r2 = clas2.shape
p_clas1 = r/len(clases)
p_clas3 = r2/len(clases)

sigma = np.cov(newl)
