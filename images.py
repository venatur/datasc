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
newl = []
for name in glob.glob('circulos_cuadrados/*.jpg'):
        print(name)
        im = Image.open(name)
        thresh = 200
        fn = lambda x : 255 if x > thresh else 0
        im = im.convert('L').point(fn)
        
        tama = np.size(im)
        dimen = tama[0]*tama[1]
        data = np.asarray(im)
        newv = np.matrix.flatten(data,'C')
       #image = Image.frombytes('RGBA', tama, im, 'raw')
        newl.append(newv)
        datax = pd.DataFrame(newl)
        datax[datax==255] = 1
        im.show()


test = 'circulos_cuadrados/test/*.jpg'

