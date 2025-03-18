import matplotlib.pyplot as plt
import numpy as np
import math

# --CONSTANTES -----------------
N = 100000   # Numero de pasos
xx0 = 0      # Posicion inicial en x
xy0 = 2      # Posicion inicial en y
v = 20       # Velocidad neta (modificar el codigo si se quiere una velocidad especifica como aqui)
angulo = 45
tau = 20     # Tiempo en segundos de la simulacion
h = tau / float(N-1)  # Paso del tiempo
g = 9.8      # Aceleracion 9.8 m/s**2

#---Obtener K ------
rho = 1.2    # Densidad del aire 
R = 6 / 100
A = math.pi * R**2  # Area 
m = 7.26     # Masa de la particula
xf = 86.74   # Distancia final 
Cd1 = 0.5    # Flujo laminar 
Cd2 = 0.75   # Flujo inestable oscilante  
Cd = [0, Cd1, Cd2]

k = np.zeros(3)
for i in range(3):
    k[i] = ((rho * A) / 2) * Cd[i]  # Constante  

# Entonces 0 es sin friccion, 1 Cd1, 2 Cd2

#-------Arreglos POSICION/ VELOCIDAD 

# Generamos un arreglo de Nx2 para almacenar posicion y velocidad
y = np.zeros((N, 2))  # La primera columna sera la posicion y la segunda la velocidad en ese mismo punto 
x = np.zeros((N, 2))  # La primera columna sera la posicion y la segunda la velocidad en ese mismo punto 

# Generamos tiempos igualmente espaciados
tiempo = np.linspace(0, tau, N)  # El inicio siempre debe ser 0 porque es asi como inicializamos el programa 

# Definicion de nuestras ecuaciones diferenciales con friccion
def EDO1(estado, tiempo, i):
    f0 = estado[1]
    f1 = -(k[i] / m) * estado[1]**2
    return np.array([f0, f1])

def EDO2(estado, tiempo, i):
    f0 = estado[1]
    if estado[1] >= 0:
        f1 = (-(k[i] / m) * estado[1]**2) - g
    else:
        f1 = ((k[i] / m) * estado[1]**2) - g
    return np.array([f0, f1])

# Metodo de Runge-Kutta
def RungeKutta(y, t, h, f, i=None):
    if i is not None:
        k1 = h * f(y, t, i)
        k2 = h * f(y + k1 / 2, t + h / 2, i)
        k3 = h * f(y + k2 / 2, t + h / 2, i)
        k4 = h * f(y + k3, t + h, i)
    else:
        k1 = h * f(y, t)
        k2 = h * f(y + k1 / 2, t + h / 2)
        k3 = h * f(y + k2 / 2, t + h / 2)
        k4 = h * f(y + k3, t + h)
    y_p = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_p

# Algoritmos utiles
t0 = tau / 2
dt = 3.e-3
err = 0.001
Nmax = 10000  # Parametros 
Err0 = 0.015
Err01 = 0.015

def Mapeo(t0, t, x, s, err): # s=0 mapeo de posiciones, s=1 mapeo de velocidades
    for j in range(N-1): 
        if abs(t[j] - t0) <= err:
            return x[j, s]
    print('\n No se encontro valor de la funcion en t=', t0)
    return 0

def bisection(a, b, err, max_iter, t, x, s):
    if Mapeo(a, t, x, s, err) * Mapeo(b, t, x, s, err) >= 0:
        raise ValueError("La funcion no cambia de signo en el intervalo dado.")
    
    iter_count = 0
    while (b - a) / 2 > err and iter_count < max_iter:
        c = (a + b) / 2  # Punto medio del intervalo
        if Mapeo(c, t, x, s, err) == 0:
            return c, iter_count
        elif Mapeo(c, t, x, s, err) * Mapeo(a, t, x, s, err) < 0:
            b = c  # La raiz esta en el subintervalo [a, c]
        else:
            a = c  # La raiz esta en el subintervalo [c, b]
        iter_count += 1

    return (a + b) / 2

def NewtonR(t, dt, x, err, Nmax, s):
    t0 = bisection(0, tau, Err0, 100, t, x, s)
    for it in range(Nmax):
        print=t0
        F = Mapeo(t0, t, x, 0, Err0)
        if F is not None and abs(F) <= err:
            break
        elif F is None:
            print('\n Valor de funcion es None en x=', t0)
            break
        df = (Mapeo(t0 + dt / 2, t, x, 0, Err0) - Mapeo(t0 - dt / 2, t, x, 0, Err0)) / dt  # Central difference 
        if df == 0:
            break
        dt = -F / df
        t0 += dt
    if it == Nmax:
        print('\n Newton no encontro raiz para Nmax=', Nmax)
    return t0

# Obtencion de funciones
def f(t, x, y, v, angulo, i): 
    y[0, 0] = xy0
    y[0, 1] = np.sin(np.radians(angulo)) * v
    x[0, 0] = xx0
    x[0, 1] = np.cos(np.radians(angulo)) * v
    for j in range(N-1): 
        x[j+1] = RungeKutta(x[j], t[j], h, EDO1, i) 
        y[j+1] = RungeKutta(y[j], t[j], h, EDO2, i)

def distancia_max(t, x, y,v, angulo,i): 
    f(t, x, y, v, angulo, i) 
    t_0 = NewtonR(t, dt, y, err, Nmax, 0)   #Buscamos el tiempo en el que  la altura se hace cero 
    x_max = Mapeo(t_0, t, x, 0, Err0)     #encontramos  que valor posee x en el tiempo to  
    return x_max  


def Vo_Xdeseada(t,dt, x, y, angulo, f, Xdeseada, err, i):  #Realmente es una adaptacion de Newton Rapson a esta funcion en especifico 
     #t0 = bisection(0, tmax, Err0, 100, t, x, s)
    v0=40
    for it in range(Nmax):
         
         F = distancia_max(t,x,y,v0,angulo,i)-Xdeseada #Hacemos este paso para lograr que en el punto de interes  el eje Y , en este caso las distiancia en x , sea cero
         
         if abs(F) <= err:
             break
         elif F is None:
             print('\n Valor de funcion es None en x=', t0)
             break
         df = (distancia_max(t,x,y,v0+dt/2,angulo,i) - distancia_max(t,x,y,v0-dt/2,angulo,i) ) / dt  # Central difference 
         
         dt = -F / df
         v0 += dt
    if it == Nmax:
         print('\n Newton no encontro raiz para Nmax=', Nmax)
    return v0


def Y_max(y):
    ymax = 0
    for j in range(N):
        if abs(y[j, 1]) < err:
            ymax = y[j, 0]
    return ymax

Velocida_champeon = np.zeros(3)
t_v = np.zeros(3)
x_max = np.zeros(3)
y_max = np.zeros(3)

xxdatos = [np.zeros(N), np.zeros(N), np.zeros(N)]
vxdatos = [np.zeros(N), np.zeros(N), np.zeros(N)]
xydatos = [np.zeros(N), np.zeros(N), np.zeros(N)]
vydatos = [np.zeros(N), np.zeros(N), np.zeros(N)]

for i in range(3):  
    if i == 0:
        print('---SIN FRICCION------')
    elif i == 1:
        print('---FRICCION C1------')
    else:
        print('---FRICCION C2------')
    
    
    Velocida_champeon[i] = Vo_Xdeseada(tiempo,1, x, y, angulo, f, xf, err, i) 
    
    
    print('La velocidad necesaria para alcanzar', xf, 'es:', Velocida_champeon[i])
    
    t_v[i] = NewtonR(tiempo, dt, y, err, Nmax, 0) # Tiempo de vuelo 
    print('El tiempo de vuelo es:', t_v[i])  
    
    x_max[i] = distancia_max(tiempo, x, y,Velocida_champeon[i],angulo,i)
    print('La distancia maxima con esta velocidad es ', x_max[i])  
    
    print('Tenemos un error relativo en la distancia de:', abs((xf - x_max[i]) / xf) * 100, '%') 
    
    y_max[i] = Y_max(y)   

    xxdatos[i] = [x[j, 0] for j in range(N)]
    vxdatos[i] = [x[j, 1] for j in range(N)]  
    
    xydatos[i] = [y[j, 0] for j in range(N)]
    vydatos[i] = [y[j, 1] for j in range(N)] 

y_Max = np.amax(y_max)

#---------------PARTE DE GRAFICACION----------------------

# Configuracion de la grafica con mejor resolucion
plt.figure(figsize=(10, 5), dpi=150)  # Tamano de la figura y resolucion DPI

# Ajustes de fuente a Courier New
plt.rc('font', family='Courier New')

# Subgrafico 1
plt.subplot(1, 2, 1)  # Subgrafico de 1 fila y 2 columnas, primer subgrafico
plt.plot(tiempo, xydatos[0], '-r', label='sin friccion')
plt.plot(tiempo, xydatos[1], '-b', label='Fricion C1')
plt.plot(tiempo, xydatos[2], '-g', label='Fricion C2')
plt.xlabel('Tiempo (s)')  # Etiqueta del eje x
plt.ylabel('Altitud (m)')  # Etiqueta del eje y
plt.title('Tiempo de Vuelo')  # Titulo de la grafica
plt.legend(loc='best')  # Mostrar leyenda en la mejor ubicacion
plt.ylim(0, 5/4*y_Max)  # Limitar el eje Y para una mejor visualizacion
plt.xlim(0, np.max(t_v))  # Limitar el eje X al rango de 0 a t_v
plt.grid(True)  # Mostrar cuadricula
plt.tight_layout()  # Ajustar el diseno para que quepa todo

# Subgrafico 2
plt.subplot(1, 2, 2)  # Subgrafico de 1 fila y 2 columnas, segundo subgrafico
plt.plot(xxdatos[0], xydatos[0], '-r', label='sin friccion')
plt.plot(xxdatos[1], xydatos[1], '-b', label='Fricion C1')
plt.plot(xxdatos[2], xydatos[2], '-g', label='Fricion C2')
plt.xlabel('Posicion en X (m)')  # Etiqueta del eje x
plt.ylabel('Altitud (m)')  # Etiqueta del eje y
plt.title('Altitud en funcion de la Posicion en X')  # Titulo de la grafica
plt.legend(loc='best')  # Mostrar leyenda en la mejor ubicacion
plt.ylim(0, 5/4*np.amax(y_max))  # Limitar el eje Y para una mejor visualizacion
plt.xlim(0, np.max(x_max))  # Limitar el eje X al rango de 0 a x_max
plt.grid(True)  # Mostrar cuadricula
plt.tight_layout()  # Ajustar el diseno para que quepa todo

#plt.suptitle('Comparacion de Trayectorias')  # Titulo general

plt.show()

