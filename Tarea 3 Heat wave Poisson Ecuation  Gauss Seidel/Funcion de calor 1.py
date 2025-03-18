import numpy as np
import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

L = 1  # longitud de la barra 
alfa = 9.786e-5    
To=100.0 #temperatura inicial 
tf=120 #tiempo final 
a = math.sqrt(alfa) 
m = 1000  # Numero de iteraciones 
N = 1000 # numero de pareticiones 
Nx = 205
Nt = 1000

# Definir la funcion
def phi1(x, t, l, c):
    X = np.zeros_like(x)
    for i in range(m):
        sM = ((-1)**i)/(2*i+1) * np.cos((2*i+1)*np.pi*x/l) * np.exp(-t*((2*i+1)*np.pi*c/l)**2)
        X += sM 
    return ((4*To)/math.pi)*X

# Crear matrices de coordenadas (x, t)
x = np.linspace(-L/2, L/2, Nx)
t = np.linspace(0, tf, Nt)
x, t = np.meshgrid(x, t)

# Calcular z utilizando la funcion definida
z = phi1(x, t, L, a) 



# Crear la figura tridimensional
fig = plt.figure(figsize=(12, 12),dpi=300)
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
surf = ax.plot_surface(x, t, z, cmap='viridis', linewidth=0, antialiased=False)



# Anadir etiquetas y titulo
ax.set_xlabel('Distancia(m)', fontsize=12)
ax.set_ylabel('Tiempo (s) ', fontsize=12)
ax.set_zlabel('Temperatura(C)', fontsize=12)
ax.set_title('                                          ', fontsize=50) 


# Ajustes para mejorar la visualizacion
ax.view_init(elev=30, azim=45)  # Cambiar la elevacion y el angulo de vision
fig.colorbar(surf, shrink=0.5, aspect=8)  # Anadir barra de color 


# Mostrar la grafica interactiva
plt.show() 
