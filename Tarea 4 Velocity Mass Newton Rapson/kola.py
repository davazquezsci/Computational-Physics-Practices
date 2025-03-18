# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:59:19 2021

@author: Prof. Uriarte
"""
from pylab import *
'''
# Definicion de constantes
N = 1000         # Numero de pasos
x0 = 0.0           # Posicion inicial
v0 = 1       # Velocidad inicial
tau =10     # Tiempo en segundos de la simulacion
h = tau / float(N-1)   # Paso del tiempo
gravedad =9.8 # Aceleracion 9.8 m/s**2
k = 3.5             # Constante elastica del resorte 
m = 0.2             # Masa de la particula

# Generamos un arreglo de Nx2 para almacenar posicion y velocidad
y = zeros([N,2])   #la primera columna sera la posicion y la segunda la velocidad en ese mismo punto 
#notemos que hablamso de una matrix de Nx2 
'''
import math
# Definicion de constantes
N = 1000 # Numero de pasos
x0 = 0.0           # Posicion inicial
v0 = 5      # Velocidad inicial
tau =10     # Tiempo en segundos de la simulacion
h = tau / float(N-1)   # Paso del tiempo
g =9.8 # Aceleracion 9.8 m/s**2
rho=1.2   #densidad del aire 
R=6/100
A= math.pi*R**2  #Area 
m = 7.26            # Masa de la particula
xf=86.74  #distancia  final 
Cd1=0.5 #flujo laminar 
Cd2=0.75 #flujo inestable osscilante 
#k = (rho*A*Cd1)/2   
k=3 
print(k)
#Hacemos una correcccion al metodo de Euler Mejorado 
# Generamos un arreglo de Nx2 para almacenar posicion y velocidad
y = zeros([N,2])   #la primera columna sera la posicion y la segunda la velocidad en ese mismo punto 
#notemos que hablamso de una matrix de Nx2 

ymas= zeros([N,2])

# tomamos los valores del estado inicial
y[0,0] = x0
y[0,1] = v0


# Generamos tiempos igualmente espaciados
tiempo = linspace(0, tau, N)

# Definimos nuestra ecuacion diferencial
def EDO(estado, tiempo):
    f0 = estado[1]  #la primera derivada de la posicion no es mas que la misma  velocidad 
    f1 = -(k/m)* (estado[1])**2
    '''
    f0 = estado[1]   #la primera derivada de la posicion no es mas que la misma  velocidad 
    f1 = -(k/m) * estado[0] - gravedad
    # se sigue la ecuacion 1  , que iguyala la derivada de la velocidad ( la aceleracion ), con la misma posicion . 
    #se hace notar ademas que el sistema esta "amortiguado " al tener la presencia de la gravedad en este mismo . 
'''

    return array([f0, f1])   


# Metodo de Euler para  resolver numericamente la EDO 
def Euler1(y, t, h, f): 
    y_s = y + h * f(y, t)  # Calculamos el valor siguiente de y
    return y_s

def Euler2(y1,y2,t,h,f):
    y_m = y1+h*((f(y1,t)+f(y2,t))/2)  #Calulamos un valor mas puro siguiente de y 
    return y_m

# Ahora calculamos!

for j in range(N-1): #Hasta N -1 por que partimos de condiciones iniciales, y por que el valor que calculamos  es para j+1 
    y[j+1] = Euler2(y[j],Euler1(y[j],tiempo[j],h,EDO) ,tiempo[j], h, EDO)

#Donde Euler se utiliza dos veces, para hacer la prediccion , de forma iterada sobre el mismo punto y luego obtener el promedio de este 


# Ahora graficamos
xdatos = [y[j,0] for j in range(N)]
vdatos = [y[j,1] for j in range(N)] 

#la forma y[j,0] se refiere a toda la primera columna , la columna 0 .

# Configuracion de la grafica con mejor resolucion
figure(figsize=(8, 6), dpi=150)  # Tamano de la figura y resolucion DPI

#Activar esta parte si se desea ver Posicion,Velocidad vs tiempo --------------------------------------------

plot(tiempo, xdatos, '-r', label='Posicion')
plot(tiempo, vdatos, '-b', label='Velocidad')
xlabel('Tiempo (s)')  # Etiqueta del eje x
ylabel('Posicion y Velocidad')  # Etiqueta del eje y

'''
#Activar esta parte si se desea ver Posicion vs Velocidad --------------------------------------------
plot(xdatos, vdatos, '-b')
xlabel('Posicion (m)')  # Etiqueta del eje x
ylabel('Velocidad (m/s)')  # Etiqueta del eje y

'''
title('                                  ')  # Titulo de la grafica
legend(loc='best')  # Mostrar leyenda en la mejor ubicacion
grid(True)  # Mostrar cuadricula
tight_layout()  # Ajustar el diseno para que quepa todo

show()