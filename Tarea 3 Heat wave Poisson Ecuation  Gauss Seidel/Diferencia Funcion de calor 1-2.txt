import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros y definiciones de la solución analítica
L = 1
alfa = 9.786e-5
To = 100.0
tf = 120
a = np.sqrt(alfa)
m = 1000
Nx = 450
Nt = 5000

def phi_analitica(x, t, l, c):
    X = np.zeros_like(x)
    for i in range(m):
        sM = ((-1)**i)/(2*i+1) * np.cos((2*i+1)*np.pi*x/l) * np.exp(-t*((2*i+1)*np.pi*c/l)**2)
        X += sM 
    return ((4*To)/np.pi)*X

x_analitica = np.linspace(-L/2, L/2, Nx)
t_analitica = np.linspace(0, tf, Nt)
x_analitica, t_analitica = np.meshgrid(x_analitica, t_analitica)
z_analitica = phi_analitica(x_analitica, t_analitica, L, a)

# Parámetros y definiciones de la solución numérica

eta = alfa*(tf/Nt)/((L/Nx)**2)

phi_numerica = np.zeros((Nx, Nt))
phi_numerica[:, 0] = To
phi_numerica[0, :] = 0
phi_numerica[-1, :] = 0

for j in range(1, Nt):
    for i in range(1, Nx-1):
        phi_numerica[i, j] = phi_numerica[i, j-1] + eta * (phi_numerica[i+1, j-1] + phi_numerica[i-1, j-1] - 2 * phi_numerica[i, j-1])

#Correccion de dimencion de la matriz  del resultado numerico 
phi_numerica=phi_numerica.transpose()

# Calcula la diferencia entre las soluciones analítica y numérica
diferencia = z_analitica - phi_numerica

# Grafica la diferencia
fig = plt.figure(figsize=(12, 12), dpi=300)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_analitica, t_analitica, diferencia ,cmap='viridis', linewidth=0, antialiased=False)

# Etiquetas y título
ax.set_xlabel('Distancia (m)', fontsize=12)
ax.set_ylabel('Tiempo (s)', fontsize=12)
ax.set_zlabel('Diferencia de Temperatura (C)', fontsize=12)
ax.set_title('                                          ', fontsize=50) 

# Mejoras en la visualización
ax.view_init(elev=30, azim=45)
fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()