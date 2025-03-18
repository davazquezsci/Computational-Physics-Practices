# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:44:07 2024

@author: Amuri
import numpy as np
"""
import numpy as np
tf = 120
alfa = 8.091e-5
L = 1 

Nt = 5000
k=L*np.sqrt(Nt/(2*alfa*tf))

print('El valor de Nx debe ser menor que:',k)