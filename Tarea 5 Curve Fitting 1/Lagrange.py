import sympy  #biblioteca que reune toda las caracteristicas dwe un sistema de algebra computacional 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  #Biblioteca con la que importamos los datos 
from pylab import *

#--------Carga de Datos ----------

data = pd.read_csv('Datos PROBLEMA 1.txt', delim_whitespace=True) #delim_whitespace=True para manejar los espacios como delimitadores.
#Asi como tenemos los datos 'i=' es tratado por 'pandas' como q
E = data.iloc[0, 1:].astype(float).values
f = data.iloc[1, 1:].astype(float).values
sigma = data.iloc[2, 1:].astype(float).values 

divNo=5

#data.iloc[1, 1:] selecciona la segunda fila (indice 1) excluyendo el primer valor (que es el encabezado de la fila). Luego, se convierte a tipo float y se extrae como un array de valores.
x=E
y=f
#--------Funciones---------------

def arg_prod(i, j):
    """ Argumento de la productoria de las bases polinomicas de 
    Lagrange.
    """
    # Variable simbolica
    x_sim = sympy.symbols('x') #aqui se hace uso de las propiedades algebraicas de sympy 
    return (x_sim-x[i]) / (x[j]-x[i]) if i != j else 1

def interpolacion_lagrage(x, y, num_puntos):
    """ Estima la curva generada por el polinomio de lagrange que 
    interpola los puntos datos
    
    args:
        x (np.array): Datos del eje x
        y (np.array): Datos del eje y
        num_puntos (int): Numero de puntos estimados a partir del polinomio
    
    returns:
        Puntos (x, y) estimados a partir del polinomio encontrado. Tupla
    """
    # Variable simbolica
    x_sim = sympy.symbols('x')
    
    # Numero de puntos ingresados
    points = len(x) #longitud del vector X 
    
    # Bases polinomicas Lj = [L1, L2, ..., Lk]
    Lj = []  #lista vacia que almacena las bases polinomicas 
    for k in range(points):
        Lk = np.prod([arg_prod(i, k) for i in range(points)])  #usa la funcion previamente definida para calcular  un porducto de elementos con np.prod
        Lj.append(Lk)  #anade el elementdo LK a la base Lj 

    # Polinomio de lagrange
    pol = sum(y*Lj)  
    
   # Aqui, y * Lj realiza un producto elemento a elemento entre los valores y y las bases Lj.
    #sum(y * Lj) suma estos productos para construir el polinomio pol.
    
    # Se generan los datos x, y a partir del polinomio encontrado
    x_test = np.linspace(min(x), max(x),num_puntos)
    y_pol = [pol.subs(x_sim, i) for i in x_test]  
    #pol.subs(x_sim, i) sustituye el valor i en la variable simbolica x_sim dentro del polinomio pol y evalua el resultado.
    
    return x_test, y_pol   

#-----------Aplicacion-----

#Obtenemos valores cada 5 Mev  
NO=int(max(x)//divNo+1) #elint es para transformar en entero el numero , el mas 1 es para incluir el 0 y el 200 


# Puntos generados por el polinomio de Lagrange
x_test, y_pol = interpolacion_lagrage(x, y,NO) 


# Grafica
# Configuracion para utilizar LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # tamano de la figura y calidad
# Datos originales
#ax.scatter(x, y, color='red', marker='o', )
# Plot de la interpolacion de Lagrange
ax.plot(x_test, y_pol, 'o', color='blue', label='Lagrange',markersize=3, linewidth=2)

plt.errorbar(E, f, yerr=sigma, fmt='or', capsize=5,label='Datos')


# Configuracion de leyenda
ax.legend(loc='best')

# Etiquetas de los ejes
ax.set_xlabel(r'$E \, (\mathrm{MeV})$', fontsize=14)
ax.set_ylabel(r'$f(E) \, (\mathrm{MeV})$', fontsize=14)

# Titulo de la grafica
ax.set_title(r'\textbf{Distribucion de $f(E)$ vs Energia de Resonancia}', fontsize=16)

# Cuadricula
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Lineas de los ejes
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Ajustar diseno para que todo quepa
fig.tight_layout()

# Mostrar la grafica
plt.show() 
#---------------------------

# Ejemplo de uso
N=2000
x_op, y_op = interpolacion_lagrage(x, y, N)  
error=0.5
E_res = x_op[np.argmax(y_op)] 


f_m = max(y_op) / 2   

def encontrar_puntos_cercanos(x_op, y_op, E_res, f_m):
    puntos_por_detras = None
    puntos_por_delante = None
    distancia_minima_por_detras = float('inf')
    distancia_minima_por_delante = float('inf')

    # Recorrer la lista de energias interpoladas
    for i in range(len(x_op)):
        # Calcular la distancia entre la energia interpolada y E_res
        distancia1 = abs(x_op[i] - E_res)
        distancia2= abs(x_op[i] - E_res)

        # Verificar si la energia interpolada esta por detras de E res
        if i < np.argmax(y_op) and distancia1 < distancia_minima_por_detras:
            if abs(f_m-y_op[i])<error:
                puntos_por_detras = x_op[i]
                distancia_minima_por_detras = distancia1
            #print(i)

        # Verificar si la energia interpolada esta por delante de E_res
        elif i > np.argmax(y_op) and distancia2 < distancia_minima_por_delante:
            if abs(f_m-y_op[i])<error:
                puntos_por_delante = x_op[i]
                distancia_minima_por_delante = distancia2
        
     

    return abs(puntos_por_detras- puntos_por_delante)

# Ejemplo de uso
gamma= encontrar_puntos_cercanos(x_op, y_op, E_res, f_m)



gammat=55 
Ert=78 

print(' Energia de resonancia', E_res,'| Error absoluto ', abs(Ert-E_res), 'Error Relativo ' , abs((Ert-E_res)/Ert))
print(' Gamma = ', gamma ,'| Error absoluto ', abs(gammat-gamma), 'Error Relativo ' , abs((gammat-gamma)/gammat))




