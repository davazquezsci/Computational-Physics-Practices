
"""
@author: D.A. Vazquez Gutierrez
"""

import math 

pi=math.pi
#Primero definimos nuestra nueva funcion seno solamente para abs(x)=<2pi
def senopirata(x):
    a=x 
    suma=a
    sumamenos=a
    amenos=0
    n=1
    while abs(a/sumamenos)>(10**(-8)): #valor del error absoluto 
        amenos=a
        sumamenos=suma
        n=n+1
        a=(((-1)*x**2)*(amenos))/(((2*n)-2)*((2*n)-1))
        suma=suma+a
    return suma
g=.25    
y=math.sin(g)
k=senopirata(g)
print("seno pirata",k, "seno original",y) 

#En este punto ya sabemos que funciona el seno pirata, ahora  arreglemos el problema numerico para  x<2pi
  
def senopirataPRIME(x):
    r = x % (2 * math.pi) #uso de modulo para obtener la generalizacion de la funcion 
    return senopirata(r)

t=45.63
u=math.sin(t)
l=senopirataPRIME(t)
print("seno pirata PRIME",l, "seno original",u)    