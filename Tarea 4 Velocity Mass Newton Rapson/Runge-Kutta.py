# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:59:19 2021

@author: Prof. Uriarte

"""
import matplotlib.pyplot as plt
from pylab import *
import math
# Definicion de constantes
N = 30000    # Numero de pasos
x0 = 0         # Posicion inicial
v0 = 100    # Velocidad inicial
tau =30  # Tiempo en segundos de la simulacion
h = tau / float(N-1)   # Paso del tiempo
g =9.8 # Aceleracion 9.8 m/s**2
rho=1.2   #§densidad del aire 
R=6/100
A= math.pi*R**2  #Area 
m = 7.26            # Masa de la particula
xf=86.74  #distancia  final 
Cd1=0.5 #flujo laminar 
Cd2=0.75 #flujo inestable osscilante 
k = (rho*A*Cd1)/2            # constante 

# Generamos un arreglo de Nx2 para almacenar posicion y velocidad
y = zeros([N,2])   #la primera columna sera la posicion y la segunda la velocidad en ese mismo punto 
#notemos que hablamso de una matrix de Nx2 

#Hacemos una correcccion al metodo de Euler Mejorado 

ymas= zeros([N,2])

# tomamos los valores del estado inicial
y[0,0] = x0
y[0,1] = v0


# Generamos tiempos igualmente espaciados
tiempo = linspace(0, tau, N)  

# Definimos nuestra ecuacion diferencial
def EDO(estado, tiempo):
    f0 = estado[1]  #la primera derivada de la posicion no es mas que la misma  velocidad 
    f1 = -(k/m)*  estado[1]**2
    # se sigue la ecuacion 1  , que iguala la derivada de la velocidad ( la aceleracion ), con la misma posicion . 
    #se hace notar ademas que el sistema esta "amortiguado " al tener la presencia de la gravedad en este mismo . 
    
    return array([f0, f1])   


# Metodo de Euler para  resolver numericamente la EDO 
def Euler1(y, t, h, f): 
    y_s = y + h * f(y, t)  # Calculamos el valor siguiente de y
    return y_s

def Euler2(y1,t,h,f):
    y_m = y1+h*((f(y1,t)+f(Euler1(y1,t,h,f),t))/2)  #Calulamos un valor mas puro siguiente de y 
    return y_m 

def RungeKutta(y, t, h, f):
    k1 = h * f(y, t)
    k2 = h * f(Euler2(y,t,h/2,f), t+h/2 )
    k3= h*f(y+k2/2,t+h/2)
    k4=h*f(y+k3,t+h)
    y_p = y + 1/6*(k1+2*k2+2*k3+k4)
    return y_p
   
'''
# Ahora calculamos!

for j in range(N-1): #Hasta N -1 por que partimos de condiciones iniciales, y por que el valor que calculamos  es para j+1 
    y[j+1] = Euler2(y[j],tiempo[j], h, EDO)

#Donde Euler se utiliza dos veces, para hacer la prediccion , de forma iterada sobre el mismo punto y luego obtener el promedio de este 
'''

#Ahora utilizamos el Algoritmo de Runge-Kutta 
for j in range(N-1): #Hasta N -1 por que partimos de condiciones iniciales, y por que el valor que calculamos  es para j+1 
    y[j+1] = RungeKutta(y[j],tiempo[j], h, EDO)


# Ahora graficamos
xdatos = [y[j,0] for j in range(N)]
vdatos = [y[j,1] for j in range(N)] 

#la forma y[j,0] se refiere a toda la primera columna , la columna 0 .

# Configuracion de la grafica con mejor resolucion
figure(figsize=(8, 6), dpi=150)  # Tamano de la figura y resolucion DPI

#Activar esta parte si se desea ver Posicion,Velocidad vs tiempo --------------------------------------------  

plt.figure(figsize=(10, 5))  # Definimos el tamaño de la figura

# Subgráfico 1
plt.subplot(1, 2, 1)  # Subgráfico de 1 fila y 2 columnas, primer subgráfico
plot(tiempo, xdatos, '-r', label='Posicion')
xlabel('Tiempo (s)')  # Etiqueta del eje x
ylabel('Posicion ')  # Etiqueta del eje y
title('                                  ')  # Titulo de la grafica
legend(loc='best')  # Mostrar leyenda en la mejor ubicacion
grid(True)  # Mostrar cuadricula
tight_layout()  # Ajustar el diseno para que quepa todo

# Subgráfico 2
plt.subplot(1, 2, 2)  # Subgráfico de 1 fila y 2 columnas, segundo subgráfico
plot(tiempo, vdatos, '-b', label='Velocidad')
xlabel('Tiempo (s)')  # Etiqueta del eje x
ylabel(' Velocidad')  # Etiqueta del eje y


title('                                  ')  # Titulo de la grafica
legend(loc='best')  # Mostrar leyenda en la mejor ubicacion
grid(True)  # Mostrar cuadricula
tight_layout()  # Ajustar el diseno para que quepa todo

show()