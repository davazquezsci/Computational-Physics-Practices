# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:45:37 2024

@author: dangv
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import math   
h=3
Lx = h*np.pi  
Ly=h*np.pi 
Nx = 1000 # Numero de particiones del eje x 
Ny = 1000 # numero de particiones  del eje y 
err=0.01  
Z=60 #iteraciones de Gauss Seidel 
p=0
def rho(x,y):
    k=np.cos(3*x+4*y)-np.cos(5*x-2*y)
    return k 

x=np.linspace(-Lx,Lx,Nx)
y=np.linspace(-Ly,Ly,Ny) 

phi = np.ones((Nx, Ny))  #Por el metodo de Gauss Seidel , hay que dar un valor "basura" a cada uno de los valores en la matriz al iniciar el proceso 


# Iteracion de la EDP 
for u in range(Z): 
    
    for j in range(1, Ny-1): #empezamos en 1 por que la ilera de t=0 ya esta inicializada 
        for i in range(1, Nx-1): #vamos de 1 a Nx-1 por que tanto el primero (0) como el ultimo ya estan incializados en x  
           
           if (y[j]%(2*np.pi))<err:
               phi[i, j] = phi[i,0]
           elif (x[i]%(2*np.pi))<err:
               phi[i, j] = phi[0,j]
           else:
              phi[i, j] = (rho(x[i],y[j])/4)+(1/4)*(phi[i+1,j]+phi[i-1,j]+phi[i,j+1]+phi[i,j-1])#Por como esta definido phi en la ecuacion 9 , hacemos un cambio de j+1 a j  
 
    
    
    p=p+1
    print((p/Z)*100,'%') 

y, x = np.meshgrid(y, x)  # Aqui t se transpone con x , nesesario para graficar correctamente la funcion hecha 
# Grafico 3D de la superficie
fig = plt.figure(figsize=(12, 12),dpi=300) #dpi=300 aumenta la calidad de  la grafica , y la proporcion de figsize hace que los labels sean de un tamano apropiado 
ax = fig.add_subplot(111, projection='3d') #el subpot 111, se refiere a que se proyecta la  grafica 1, de un total de graficas 1x1 , hay mas configuraciones 212, 223, etc 
surf=ax.plot_surface(y, x, phi, cmap='viridis', linewidth=0, antialiased=False)  
# Anadir etiquetas y titulo
ax.set_xlabel('Distancia X (m)', fontsize=12)
ax.set_ylabel('Distancia Y (m) ', fontsize=12)
ax.set_zlabel('Funcion', fontsize=12)
ax.set_title('                                          ', fontsize=50) 

# Ajustes para mejorar la visualizacion
ax.view_init(elev=45, azim=30)  # Cambiar la elevacion y el angulo de vision
fig.colorbar(surf, shrink=0.5, aspect=8)  # Anadir barra de color  


plt.show()