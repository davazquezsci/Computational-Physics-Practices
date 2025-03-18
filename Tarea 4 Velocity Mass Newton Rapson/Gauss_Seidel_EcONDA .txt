# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:45:37 2024

@author: dangv
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import math    

#Biblioteca para animar
import matplotlib.animation as anim
from PIL import Image
from glob import glob


Lx = 2   #maximo de distancia espacial 
Lt=20  #maximo de distancia espacial 
Nx = 500 # Numero de particiones del eje x 
Nt =8000 # numero de particiones  del eje y  
dx=Lx/Nx
dt=Lt/Nt
F=5 #Fuerza de tension de la cuerda en x
M=3 #Densidad de masa lineal 
cf=math.sqrt(5/3) #velocidad de fase
cm=dx/dt
CNT=cf/cm

print('Velocidad de Fase:',cf,'\n Velocidad de Malla',cm,'\n Condicion de Courant',CNT)
 

err=0.01  
Z=60 #iteraciones de Gauss Seidel 
p=0 #Sirve para medir el numero de iteraciones en la funcion para Gauss Seidel  
x=np.linspace(0,Lx,Nx)
t=np.linspace(0,Lt,Nt) 

y = np.ones((Nx, Nt))  #Por el metodo de Gauss Seidel , hay que dar un valor "basura" a cada uno de los valores en la matriz al iniciar el proceso 

# Definimos las ecuaciones de las condiciones de frontera 

def Condicion_Inicial(x): 
    return math.cos((3*math.pi) * x/Lx)

def Condicion_Frontera_L0(t):
    return 0

def Condicion_Frontera_Lx(t):
    return 0
    
# Inicializacion de las condiciones de frontera e iniciales  

for j in range(Nt):
    y[0, j] = Condicion_Frontera_L0(t[j])
    y[Nx-1, j] = Condicion_Frontera_Lx(t[j])

for i in range(Nx):
    y[i, 0] = Condicion_Inicial(x[i]) 
    
print('Generando funcion...')

# Iteracion de la EDP 
for u in range(Z): 
    for j in range(1, Nt-1): 
        for i in range(1, Nx-1):
            #y[i, j] = (1 / (CNT**2 + 2)) * ((CNT**2) * (y[i+1, j] + y[i-1, j]) + y[i, j+1] + y[i, j-1])
            y[i,j+1]=2*y[i, j]-y[i,j-1]+(CNT**2)*(y[i+1, j] + y[i-1, j]-2*y[i, j])
            
          
    
    p += 1
    print((p / Z) * 100, '%') 

print('Graficando...')

# Graficacion ----------------------
n=40  #Numero de  diviciones 
s=Nt//n
for i in range(n):
    d=s*i
    y1=x
    for j in range(Nx):
        y1[j]=y[j,d]
    
    plt.figure(figsize=(12, 6), dpi=300)  # dpi=300 aumenta la calidad de la gráfica
    plt.plot(x, y1, color='b', label=f'Tiempo t={t[i]:.2f}s')

    # Añadir etiquetas y título
    plt.xlabel('Distancia X (m)', fontsize=12)
    plt.ylabel('Amplitud', fontsize=12)
    plt.title('Funcion de Onda', fontsize=20)
    plt.legend()
    
    # Ajustes para mejorar la visualización
    plt.ylim(-350, 350)  # Limitar el eje Y para una mejor visualización de la función seno
    plt.xlim(0, 2)  # Limitar el eje X al rango de 0 a 2

    # Guardar la figura
    plt.savefig(f'figura_{i:03d}.png')  # Guardar con un nombre único para cada frame

    # Cerrar la figura para evitar sobrecarga de memoria
    plt.close() 
    
    
# Animacion ------------------ 

# Obtener la lista de archivos que coinciden con el patrón
files = sorted(glob('figura_0*.png'))

# Verificar si se encontraron archivos
if not files:
    print("No se encontraron archivos de imagen para animar.")
else:
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')  # Ocultar los ejes

    # Lista para almacenar los objetos de imagen
    ims = []

    # Cargar cada imagen y añadirla a la lista de frames
    for fname in files:
        im = ax.imshow(Image.open(fname), animated=True)
        ims.append([im])

    # Crear la animación
    ani = anim.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)

    # Guardar la animación como un archivo GIF
    ani.save('animacion.gif', writer='pillow')

    plt.close()  # Cerrar la figura después de guardar