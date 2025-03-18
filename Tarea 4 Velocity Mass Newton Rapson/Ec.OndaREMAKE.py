import os
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import animation as anim
from PIL import Image
from glob import glob

# Variables generales
Lx = 2 # Máximo de distancia espacial
Lt = 15  # Máximo de tiempo
Nx = 50  # Número de particiones del eje x
Nt = 500  # Número de particiones del eje t
dx = Lx / (Nx-1)
dt = Lt / Nt
F = 3  # Fuerza de tensión de la cuerda en x
M = 4  # Densidad de masa lineal
cf = math.sqrt(F / M)  # Velocidad de fase
cm = dx / dt
CNT = cf / cm

print('Velocidad de Fase:', cf, '\nVelocidad de Malla:', cm, '\nCondicion de Courant:', CNT)

# Verificar condición de Courant
if CNT > 1:
    raise ValueError("La condición de Courant no se cumple. Ajusta dx o dt para que cf/cm <= 1.")

x = np.linspace(0, Lx, Nx)
t = np.linspace(0, Lt, Nt)

y = np.zeros((Nx, Nt))  # Inicializa la matriz y con valores "basura"

# Definir las condiciones de frontera e inicial
def Condicion_Inicial(x): 
    return math.sin((3 * math.pi) * x / Lx)

def Condicion_Frontera_L0(t):
    return 0

def Condicion_Frontera_Lx(t):
    return 0

# Aplicar las condiciones de frontera e inicial
for j in range(Nt):
    y[0, j] = Condicion_Frontera_L0(t[j])
    y[Nx-1, j] = Condicion_Frontera_Lx(t[j])

for i in range(Nx):
    y[i, 0] = Condicion_Inicial(x[i])

print('Generando función...')

# Iteración de la EDP
for j in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        y[i, j + 1] = 2 * y[i, j] - y[i, j - 1] + (CNT**2) * (y[i + 1, j] + y[i - 1, j] - 2 * y[i, j])

# Visualización de los resultados
fig, ax = plt.subplots()
line, = ax.plot(x, y[:, 0])

def animate(i):
    line.set_ydata(y[:, i])
    return line,

ani = anim.FuncAnimation(fig, animate, frames=Nt, interval=20, blit=True)
plt.show()
