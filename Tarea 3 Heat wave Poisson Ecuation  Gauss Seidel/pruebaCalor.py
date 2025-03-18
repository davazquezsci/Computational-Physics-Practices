# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:10:03 2024

@author: Amuri
"""
import math
import pandas as pd  



def prueba(x):
    a1=0
    a2=0
    suma=0
    m=0
    L=3
    for i in range(6000):
        l=2*m+1
        a1=((-1)**(m))/(l)
        suma=suma+((400)/(math.pi))*a1*math.cos(l*math.pi*x/L)
        m=m+1
    print('el valor es',suma)
        
rock=prueba(6)

print('l valor es',rock)
        