import numpy as np
import scipy.special as ss 
import matplotlib.pyplot as plt
from pylab import *


x_values = [0.1, 1, 10] 
l_values = [3,5,8]

# Ademas queremos que tenga una profundidad para los valores Up , Los Down , los Reales  , Error up y error down , 5 en total
tensor=np.zeros((len(x_values),len(l_values),5))

w=35

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
    if l == w:
        return 0
    elif l == w-1:
        return 1
    else:
        return ((2*l +3 ) / x) * bessel_down_a(l+1, x) - bessel_down_a(l+2, x)
"""
 Primeraparte  de algoritmo de miller  conseguimos  la version con errores 
 para despues conseguir una normalizacion  respecto a la base  jC0
 """   
def bessel_down(l, x):
    if l == w:
        return 0*((np.sin(x) / x)/bessel_down_a(0,x))
    elif l == w-1:
        return 1*((np.sin(x) / x)/bessel_down_a(0,x))
    else:
        return (((2*l +3) / x) * bessel_down_a(l+1, x) - bessel_down_a(l+2, x))*((np.sin(x) / x)/bessel_down_a(0,x))
"""
Se anade el ajuste para el down
"""


for i, x in enumerate(x_values):
    for j, l in enumerate(l_values):
        for m in range(5):
            if m == 0:
                tensor[i, j, m] = bessel_up(l, x)
            elif m == 1:
                tensor[i, j, m] = bessel_down(l, x)
            elif m == 2:
                tensor[i, j, m] = ss.spherical_jn(l, x)
            elif m == 3:
                tensor[i, j, m] = abs((tensor[i, j, m-3] - tensor[i, j, m-1]) / tensor[i, j, m-1])
            else:
                tensor[i, j, m] = abs((tensor[i, j, m-3] - tensor[i, j, m-2]) / tensor[i, j, m-2])
    
            
            
#Escribe y compara los mismos balores de la ecuacion esferica de bessel up y down 
#print(f"Error_relativo_{l}({x}) (Up): {Error_up_l}, Error_relativo_{l}({x}) (Down): {Error_down_l} ") 
#print(f"J_{l}({x}) (Up): {j_l_up}, J_{l}({x}) (Down): {j_l_down} ") 


#-----------Aplicacion-----


# Configuración para utilizar LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Crear subgráficas
fig, axs = plt.subplots(3, 1, figsize=(8, 18), dpi=150)  # Tres subgráficas en una columna

for idx,x in enumerate(x_values):
    ax = axs[idx]
    ax.plot(l_values, tensor[ idx,:, 0], '-', color='blue', label='Bessel up', markersize=3, linewidth=2)
    ax.plot(l_values, tensor[ idx,:, 1], '-', color='red', label='Bessel down', markersize=3, linewidth=2)
    ax.plot(l_values, tensor[ idx,:, 2], '-', color='green', label='Bessel real', markersize=3, linewidth=2)
    
    # Configuración de leyenda
    ax.legend(loc='best')
    
    # Etiquetas de los ejes
    ax.set_xlabel(r'$ l $', fontsize=14)
    ax.set_ylabel(rf'$Bessel_{{(x={x})}}$', fontsize=14)
    
    # Título de la gráfica
    ax.set_title(rf'\textbf{{Bessel Function for $x={x}$}}', fontsize=16)
    
    # Cuadrícula
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Líneas de los ejes
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

# Ajustar diseño para que todo quepa
fig.tight_layout()

# Mostrar la gráfica
plt.show()