# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:49:50 2024

@author: dangv
"""

import matplotlib.pyplot as plt
from pylab import *
import math

N = 100000   # Numero de pasos
x10 = 2        # Posicion inicial en masa 1 k lineal 
v10 = 0    # Velocidad inicial en en masa 1 k lineal 
x20 = 0        # Posicion inicial en masa 2 k lineal 
v20 = 0   # Velocidad inicial en masa 2 k lineal 
x30 = 2        # Posicion inicial en masa 1 k no lineal
v30 = 0   # Velocidad inicial masa 1 k no lineal
x40 = 0        # Posicion inicial masa 2 k no lineal
v40 = 0   # Velocidad inicial en masa 2 k no lineal
k1=0.3 #Constante resortes laterales
k=0.75 #Constante resorte intermedio 
tau =40  # Tiempo en segundos de la simulacion
h = tau / float(N-1)   # Paso del tiempo
m = 1.2          # Masa de la particula


# Generamos un arreglo de Nx2 para almacenar posicion y velocidad
x1 = zeros([N,2])   #la primera columna sera la posicion y la segunda la velocidad en ese mismo punto 
x2 = zeros([N,2])   #la primera columna sera la posicion y la segunda la velocidad en ese mismo punto 
#notemos que hablamso de una matrix de Nx2 
x3 = zeros([N,2]) 
x4 = zeros([N,2])

# tomamos los valores del estado inicial
x1[0,0] = x10
x1[0,1] = v10 

x2[0,0] = x20
x2[0,1] = v20  

x3[0,0] = x30
x3[0,1] = v30   

x4[0,0] = x40
x4[0,1] = v40  


#Tomamos el cambio de variable para encontrar soluciones armonicas simples.
xa=x1+x2   #Estado armonico 1 fase lineal
xb=x2-x1  #Estado armonico 2 no fase  lineal

xc=x3+x4   #Estado armonico 1 fase nolineal
xd=x4-x3  #Estado armonico 2 no fase no lineal



# Generamos tiempos igualmente espaciados
tiempo = linspace(0, tau, N)   #el inicio siempre debe ser 0 por que es asi como inicializamos el programa 

#------------Definicon de nuestras ecuaciones diferenciales con constante k lineal  ------------------------

def EDO1(estado1, tiempo):
    f0 = estado1[1]  #la primera derivada de la posicion no es mas que la misma  velocidad 
    f1 = -(k1/m)*estado1[0] 
    # se sigue la ecuacion 1  , que iguala la derivada de la velocidad ( la aceleracion ), con la misma posicion . 
    #se hace notar ademas que el sistema esta "amortiguado " al tener la presencia de la gravedad en este mismo . 
    
    return array([f0, f1])  

def EDO2(estado1, tiempo):
    f0 = estado1[1]  #la primera derivada de la posicion no es mas que la misma  velocidad 
    f1 = -((k1+2*k)/m)*estado1[0] 
    # se sigue la ecuacion 1  , que iguala la derivada de la velocidad ( la aceleracion ), con la misma posicion . 
    #se hace notar ademas que el sistema esta "amortiguado " al tener la presencia de la gravedad en este mismo . 
    
    return array([f0, f1])   

#------------Definicon de nuestras ecuaciones diferenciales con constante k  NO lineal  ------------------------

def EDO3(estado1, tiempo):
    f0 = estado1[1]  #la primera derivada de la posicion no es mas que la misma  velocidad 
    f1 = -(k1/m)*(estado1[0]+0.1*estado1[0]**3)
    # se sigue la ecuacion 1  , que iguala la derivada de la velocidad ( la aceleracion ), con la misma posicion . 
    #se hace notar ademas que el sistema esta "amortiguado " al tener la presencia de la gravedad en este mismo . 
    
    return array([f0, f1])  

def EDO4(estado1, tiempo):
    f0 = estado1[1]  #la primera derivada de la posicion no es mas que la misma  velocidad 
    f1 = -((k1+2*k)/m)*(estado1[0]+0.1*estado1[0]**3)
    # se sigue la ecuacion 1  , que iguala la derivada de la velocidad ( la aceleracion ), con la misma posicion . 
    #se hace notar ademas que el sistema esta "amortiguado " al tener la presencia de la gravedad en este mismo . 
    
    return array([f0, f1])  


#---------------- Metodos para obtener las funciones de las ecuaciones diferenciales ---------

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

#----------------------- Algoritmos Utiles ---------------------------------

t0 = tau / 2
dt = 3.e-3
err = 0.001
Nmax = 100 # Parametros 
Err0=0.0015
Err01=0.015

def Mapeo(t0, t, x, s, err):
    for j in range(N-1): 
        #print('Iteracion',j,'diferencia:',abs(t[j] - t0))
        if abs(t[j] - t0) <= err:
            F = x[j, s]
            return F
    print('\n No se encontro valor de la funcion en t=', t0)
    return 0

#Usaremos el metodo de la biseccion para tener una aproximacion a la raiz y ya despues purificamos con newton rapson 
def bisection( a, b, err, max_iter,t,x,s):
    """
    Implementacion del metodo de biseccion para encontrar una raiz de una funcion.

    Args:
        f: Funcion cuya raiz se busca.
        a: Extremo izquierdo del intervalo inicial.
        b: Extremo derecho del intervalo inicial.
        tol: Tolerancia para el error absoluto entre iteraciones consecutivas.
        max_iter: Numero maximo de iteraciones permitidas.

    Returns:
        float: Aproximacion de la raiz encontrada.
        int: Numero de iteraciones realizadas.
    
    # Verificar si la raiz esta dentro del intervalo [a, b]
    #como esta funcion oscila no tiene sentido agregar esto 
    if Mapeo(a, t, x, s, err) * Mapeo(b, t, x, s, err)  >= 0:
        print(Mapeo(a, t, x, s, err) * Mapeo(b, t, x, s, err))
        raise ValueError("La funcion no cambia de signo en el intervalo dado.")
"""
    # Inicializar variables
    iter_count = 0
    while (b - a) / 2 > err and iter_count < max_iter:
        c = (a + b) / 2  # Punto medio del intervalo
        if Mapeo(c, t, x, s, err) == 0:
            return c, iter_count
        elif Mapeo(c, t, x, s, err)* Mapeo(a, t, x, s, err)< 0:
            b = c  # La raiz esta en el subintervalo [a, c]
        else:
            a = c  # La raiz esta en el subintervalo [c, b]
        iter_count += 1

    return (a + b) / 2

def NewtonR(t, dt, x, Err0, Nmax,s,a,b):
    t0=bisection( a, b, Err01,100,t,x,s)
    for it in range(0, Nmax + 1):
        F = Mapeo(t0, t, x, 0 ,Err0)
        if F is not None and abs(F) <= err:  
            break
        elif F is None:
            print('\n Valor de funcion es None en x=', t0)
            break
       # print('Iteracion=', it, 'x=', t0, 'f(x)=', F)
        df = (Mapeo(t0 + dt / 2, t, x, 0,Err0) - Mapeo(t0 - dt / 2, t, x, 0,Err0)) / dt  # Central difference 
        if df == 0:
            # Division por cero en df. No se puede continuar.
            break
        dt = -F / df
        t0 += dt  # Nueva propuesta
    if it == Nmax + 1:
        #print('\n Newton no encontro raiz para Nmax=', Nmax)
        return None 
    return t0 
#El algoritmo de newton es preciso , pero nesesita  un valor inicial cercano a la raiz para ser realmente efectivo , por eso usaremos en conjunto biseccion  y newton rapson 
#-----------------------Obtencion de  funciones ----------------------

def f(t, xa, xb): 
    for j in range(N-1): 
        xa[j+1] = RungeKutta(xa[j], t[j], h, EDO1) 
        xb[j+1] = RungeKutta(xb[j], t[j], h, EDO2) 
    
def n(t, xa,xb): 
    for j in range(N-1): 
        xa[j+1] = RungeKutta(xa[j], t[j], h, EDO3)
        xb[j+1] = RungeKutta(xb[j], t[j], h, EDO4) 
        
f(tiempo,xa,xb) 
n(tiempo,xc,xd)


#Utilizaremos las funciones de biseccion y  Newton Rapson  para encontrar
#donde la distancia entre raices, "Un medio periodo "         
#Obtenemos las posicones originales en funcion de las del cambio de variable 

def FrecAng(t, dt, x, err,err1, Nmax,s):   
    r0=0
    r1=NewtonR(t, dt, x, err, Nmax,s,r0,tau) 
    iter_count=0
    while abs(r1-r0)>err:
       iter_count=+1
       r0=r1
       r1=NewtonR(t, dt, x, err, Nmax,s,0,r0) 
       #print('ro',iter_count,'=',r0)
#entonces r0 la raiz mas cercana al cero 
    r1=0
    r2=NewtonR(t, dt, x, err, Nmax,s,r0+err1,tau) 
    iter_count=0
    while abs(r2-r1)>err:
       iter_count=+1
       r1=r2
       r2=NewtonR(t, dt, x, err, Nmax,s,r0+err1,r1) #el anadir el error en la cota inferior de la busqueda de la raiz evita que vuelva a determinarse en la  raiz 
       #print('r1',iter_count,'=',r1)
       
    SP=r1-r0 #Semiperiodo 
    FA=np.pi/SP

    return FA   

#Los convertimos a datos las variables-------------------------------
x1=(xa+xb)/2 
x2=(xa-xb)/2  


xadatos = [xa[j, 0] for j in range(N)]
vadatos = [xa[j, 1] for j in range(N)] 

xbdatos = [xb[j, 0] for j in range(N)]
vbdatos = [xb[j, 1] for j in range(N)] 
 
x1datos = [x1[j, 0] for j in range(N)]
v1datos = [x1[j, 1] for j in range(N)] 

x2datos = [x2[j, 0] for j in range(N)]
v2datos = [x2[j, 1] for j in range(N)] 

x3datos = [x3[j, 0] for j in range(N)]
v3datos = [x3[j, 1] for j in range(N)] 

x4datos = [x4[j, 0] for j in range(N)]
v4datos = [x4[j, 1] for j in range(N)] 


#GRAFICACION -----------------------------------------------

# Definimos el tamano de la figura
plt.figure(figsize=(12, 6))

# Ajustes de fuente a Courier New
plt.rc('font', family='Courier New')

# Subgrafico 1
plt.subplot(1, 2, 1)  # Subgrafico de 1 fila y 2 columnas, primer subgrafico
plt.plot(tiempo, x1datos, '-r', label='Masa 1')
plt.xlabel('Tiempo (s)', fontsize=12)  # Etiqueta del eje x
plt.ylabel('Amplitud (m)', fontsize=12)  # Etiqueta del eje y
plt.title('Amplitud masa 1 respecto a t', fontsize=14)  # Titulo de la grafica
plt.legend(loc='best', fontsize=10)  # Mostrar leyenda en la mejor ubicacion
plt.grid(True)  # Mostrar cuadricula

# Subgrafico 2
plt.subplot(1, 2, 2)  # Subgrafico de 1 fila y 2 columnas, segundo subgrafico
plt.plot(tiempo, x2datos, '-b', label='Masa 2')
plt.xlabel('Tiempo (s)', fontsize=12)  # Etiqueta del eje x
plt.ylabel('Amplitud (m)', fontsize=12)  # Etiqueta del eje y
plt.title('Amplitud masa 2 respecto a t', fontsize=14)  # Titulo de la grafica
plt.legend(loc='best', fontsize=10)  # Mostrar leyenda en la mejor ubicacion
plt.grid(True)  # Mostrar cuadricula

# Ajustar el diseno para que quepa todo
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Titulo general
plt.suptitle('Comparacion de movimiento de coiladores acoplados : x1o!=x2o=0', fontsize=16, fontfamily='Courier New')

# Mostrar la grafica
plt.show()  



#Comparacion Valores Numericos con teoricos ------------- 


wat=math.sqrt(k1/m)
wbt=math.sqrt((k1+2*k)/m)  

#Notamos que en los cambios de variable , para cuando X1=X2 o X1=-X2, entonces los cambios de 
#variable , una de las dos se hacen cero , entonces las frecuencias angulares de alguna de ellas se hace cero
#Para evitar conflictos y modificar la funcion mapeo , entonces aplicamos unas funciones if
#SIN EMBARGO, lo siguiente no tiene tento sentido fue los casos especiales, ya que 
#en los casos especiales, las ocndiciones iniciales harian que se haga cero  alguna 
#alguno de los cambios de variable 

if x10==0 & x20==0 : 
    wan=0
    wbn=0 
elif x10==-x20:
    wan=0
    wbn=FrecAng(tiempo,dt,xb,Err0,Err01,Nmax,0) 
elif x10==x20: 
    wan=FrecAng(tiempo,dt,xa,Err0,Err01,Nmax,0) 
    wbn=0 
else :
    wbn=FrecAng(tiempo,dt,xb,Err0,Err01,Nmax,0) 
    wan=FrecAng(tiempo,dt,xa,Err0,Err01,Nmax,0) 
    
if x30==0 & x40==0 : 
    wcn=0
    wdn=0 
elif x30==-x40:
    wcn=0
    wdn=FrecAng(tiempo,dt,xd,Err0,Err01,Nmax,0) 
elif x30==x40: 
    wcn=FrecAng(tiempo,dt,xc,Err0,Err01,Nmax,0) 
    wdn=0 
else :
    wdn=FrecAng(tiempo,dt,xd,Err0,Err01,Nmax,0) 
    wcn=FrecAng(tiempo,dt,xc,Err0,Err01,Nmax,0) 
    
    


print('\n Frecuencias Normales con K lineal :\n')

print('fase: Teorica/Numerica',wat,'/',wan)
print('contrafase: Teorica/Numerica',wbt,'/',wbn)    


print('\n Frecuencias Normales con K no lineal :\n')

print('fase:Numerica',wcn)
print('contrafase : Numerica',wdn)

print('\n Veamos entonces las diferencias entre las frecuencias de los modos normales: \n ')

if wan==0 or wdn==0:
    print('fase:Numerica  Error absoluto ',abs(wan-wcn))
    print('Contrafase :Numerica  Error absoluto ',abs(wbn-wdn))    
else:
    print('fase:Numerica  Error absoluto ',abs(wan-wcn),'Error relativo :',abs((wan-wcn)/wan),'\n')
    print('Contrafase :Numerica  Error absoluto ',abs(wbn-wdn),'Error relativo :',abs((wbn-wdn)/wdn))    

