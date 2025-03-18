# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:45:37 2024

@author: dangv
"""
#Biblioteca para poder manejar archivos (Se usa en la parte de crear carpetas para ingresar imagenes de generacion de la animacion)
import os

#Biblioteca de matematicas, vectores y  graficos 

import numpy as np
import matplotlib.pyplot as plt
import math

#Biblioteca para animaciones 
from matplotlib import animation as anim
from PIL import Image
from glob import glob

#Variables generales 

Lx = 2  # Maximo de distancia espacial 
Lt = 20  # Maximo de tiempo
Nx = 1000  # Numero de particiones del eje x 
Nt = 8000  # Numero de particiones del eje t  
dx = Lx / Nx
dt = Lt / Nt
F = 5  # Fuerza de tension de la cuerda en x
M = 3  # Densidad de masa lineal 
cf = math.sqrt(F / M)  # Velocidad de fase
cm = dx / dt
CNT = cf / cm

print('Velocidad de Fase:', cf, '\nVelocidad de Malla:', cm, '\nCondicion de Courant:', CNT)

Z = 60  # Iteraciones de Gauss-Seidel
x = np.linspace(0, Lx, Nx)
t = np.linspace(0, Lt, Nt)

y = np.ones((Nx, Nt))  # Inicializa la matriz y con valores "basura"

# Definimos las ecuaciones de las condiciones de frontera -----


def Condicion_Inicial(x): 
    return math.sin((3 * math.pi) * x / Lx)  #Funion de condicion inicial IMPORTANTE que la condicion  inicial y las condiciones de frontera concuerden 

def Condicion_Frontera_L0(t):
    return 0

def Condicion_Frontera_Lx(t):
    return 0

# Inicializacion de las condiciones de frontera e iniciales  ---------


for j in range(Nt):
    y[0, j] = Condicion_Frontera_L0(t[j])
    y[Nx-1, j] = Condicion_Frontera_Lx(t[j])

for i in range(Nx):
    y[i, 0] = Condicion_Inicial(x[i]) 

print('Generando funcion...')

# Iteracion de la EDP 
for u in range(Z):
    for j in range(Nt - 1):
        for i in range(1, Nx - 1):  # Corrige los limites para no sobrescribir las fronteras
            if j == 0:
                y[i, 1] = y[i, 0] +  0.5*(CNT**2) * (y[i + 1, 0] + y[i - 1, 0] - 2 * y[i, 0]) #Pequeno paso  que es util matematicamente  por la falta de informacion en y[i,j-1], cuando j==0
            else:
                y[i, j + 1] = 2 * y[i, j] - y[i, j - 1] + (CNT**2) * (y[i + 1, j] + y[i - 1, j] - 2 * y[i, j])
    print((u + 1) / Z * 100, '%')

print('Graficando...')

# Crear la carpeta "imagenes_grafica" si no existe
output_dir = "imagenes_grafica"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Crear las figuras para cada instante de tiempo
n = 100  # Numero de divisiones
s = Nt // n
for i in range(n):
    d = s * i
    y1 = y[:, d]  # Obtener la columna correspondiente del tiempo t = d * dt
    
    plt.figure(figsize=(12, 6), dpi=300)  # dpi=300 aumenta la calidad de la grafica
    plt.plot(x, y1, color='b', label=f'Tiempo t={t[d]:.2f}s')

    # Anadir etiquetas y titulo
    plt.xlabel('Distancia X (m)', fontsize=12)
    plt.ylabel('Amplitud', fontsize=12)
    plt.title('Funcion de Onda', fontsize=20)
    plt.legend()
    
    # Ajustes para mejorar la visualizacion
    plt.ylim(-1.1, 1.1)  # Limitar el eje Y para una mejor visualizacion de la funcion seno
    plt.xlim(0, Lx)  # Limitar el eje X al rango de 0 a Lx

    # Guardar la figura
    plt.savefig(f'{output_dir}/figura_{i:03d}.png')  # Guardar con un nombre unico para cada frame

    # Cerrar la figura para evitar sobrecarga de memoria
    plt.close()
    
print('Imagenes generadas exitosamente\n Generando GIF...')

# Animacion ------------------ 

# Obtener la lista de archivos que coinciden con el patron
files = sorted(glob(f'{output_dir}/figura_*.png'))

# Verificar si se encontraron archivos
if not files:
    print("No se encontraron archivos de imagen para animar.")
else:
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')  # Ocultar los ejes

    # Lista para almacenar los objetos de imagen
    ims = []

    # Cargar cada imagen y anadirla a la lista de frames
    for fname in files:
        im = ax.imshow(Image.open(fname), animated=True)
        ims.append([im])

    # Crear la animacion
    ani = anim.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)

    # Guardar la animacion como un archivo GIF
    ani.save('animacion.gif', writer='pillow')

    plt.close()  # Cerrar la figura despues de guardar
    
print('GIF generado exitosamente')
