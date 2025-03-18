import matplotlib.pyplot as plt

from pylab import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from math import cos

x0 = 30
dx = 3.e-4
err = 0.001
Nmax = 100  # Parámetros
L = x0 + 30
Nx = 1000

def f(x):
    return 2 * np.cos(x) - x  # Función

def NewtonR(x, dx, err, Nmax):
    for it in range(0, Nmax + 1):
        F = f(x)
        if abs(F) <= err:  # ¿Es raíz dentro del error?
            print('\n Raíz encontrada, f(raíz)=', F, ', Raíz=', x, ', error=', err)
            break
        print('Iteracion=', it, 'x=', x, 'f(x)=', F)
        df = (f(x + dx / 2) - f(x - dx / 2)) / dx  # Central difference
        dx = -F / df
        x += dx  # Nueva propuesta
    if it == Nmax + 1:
        print('\n Newton no encontró raíz para Nmax=', Nmax)
    return x

raiz = NewtonR(x0, dx, err, Nmax)

x = np.linspace(-L, L, Nx)
y = f(x)

# Graficar la función
fig = plt.figure(figsize=(6, 6),dpi=300)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gráfico de la función')

# Marcar la raíz encontrada
plt.scatter(raiz, f(raiz), color='red', label='Raíz encontrada')
plt.legend()
legend(loc='best')  # Mostrar leyenda en la mejor ubicacion
grid(True)  # Mostrar cuadricula
tight_layout()  # Ajustar el diseno para que quepa todo

plt.show()
