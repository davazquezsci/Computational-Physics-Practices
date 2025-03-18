import numpy as np

def bessel_up(l, x):
    if l == 0:
        return np.sin(x) / x
    elif l == 1:
        return (np.sin(x) / x**2) - (np.cos(x) / x)
    else:
        return ((2*l - 1) / x) * bessel_up(l-1, x) - bessel_up(l-2, x) 
"""
Se anade el ajuste para el up 
"""
def bessel_down_a(l, x):
    if l == 24:
        return 0
    elif l == 23:
        return 1
    else:
        return ((2*l +3 ) / x) * bessel_down_a(l+1, x) - bessel_down_a(l+2, x)
"""
 Primeraparte  de algoritmo de miller  conseguimos  la version con errores 
 para despues conseguir una normalizacion  respecto a la base  jC0
 """   
def bessel_down(l, x):
    if l == 24:
        return 0*((np.sin(x) / x)/bessel_down_a(0,x))
    elif l == 23:
        return 1*((np.sin(x) / x)/bessel_down_a(0,x))
    else:
        return (((2*l +3) / x) * bessel_down_a(l+1, x) - bessel_down_a(l+2, x))*((np.sin(x) / x)/bessel_down_a(0,x))
"""
Se anade el ajuste para el down
"""

x_values = [0.1, 1, 10]
for x in x_values:
    print(f"Para x = {x}:")
    for l in range(25):
        j_l_up = bessel_up(l, x)
        j_l_down = bessel_down(l, x) 
        Error_l= abs(j_l_up-j_l_down)/(abs(j_l_up)+abs(j_l_down))
        print(f"J_{l}({x}) (Up): {j_l_up}, J_{l}({x}) (Down): {j_l_down} , Error_{l} :{Error_l}") 
#Escribe y compara los mismos balores de la ecuacion esferica de bessel up y down 