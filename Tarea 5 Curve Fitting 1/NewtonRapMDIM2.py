import matplotlib.pyplot as plt
from pylab import * 
import numpy as np
import numpy.linalg as npm
import math  
import pandas as pd  #Biblioteca con la que importamos los datos 

#--------Carga de Datos ----------

data = pd.read_csv('Datos PROBLEMA 1.txt', delim_whitespace=True) #delim_whitespace=True para manejar los espacios como delimitadores.
#Asi como tenemos los datos 'i=' es tratado por 'pandas' como q
E = data.iloc[0, 1:].astype(float).values
fo = data.iloc[1, 1:].astype(float).values
sigma = data.iloc[2, 1:].astype(float).values 
sigma=sigma

# Definicion de  funciones f que minimizan \chi^2   --------------

#data.iloc[1, 1:] selecciona la segunda fila (indice 1) excluyendo el primer valor (que es el encabezado de la fila). Luego, se convierte a tipo float y se extrae como un array de valores.
x=E
y=fo


N=3  #dimencion  del problema a resolvedr  
n=400 #Numero de espacios  de los parametros 

a_min = [50000, 10, 10]  # Maximos y minimos de cada  dimencion  de a_i , CAMBIAR SI AUMENTAN LOS PARAMETOS N !!!
a_max = [100000, 100, 1000]  #ojo con los parametros de a_n , ya que hay que evitar indeterminaciones en las matrices 

a = []

for i in range(N):
    a.append(np.linspace(a_min[i], a_max[i], n))

a = np.meshgrid(*a)   #generalizacion para cualquier n de a[0],a[1],a[2]= np.meshgrid(a[0],a[1],a[2])


#-----Eleccion de Semilla ----------

Ra = []

for i in range(N):
    Ra.append(a_max[i] - a_min[i])
Ra=np.array(Ra)

ta = []

for i in range(N):
    ta.append(Ra[i]/2) 

ta=np.array(ta)   #semilla tentativa 

Tx=[80400,70,560]  

ta=Tx     #EL ERROR RADICABA EN LA SEMILLA ELEGIDA !!!!!  estaba obteniendo un buen resultado , para ENERGIAS NEGATIVAS!!!!!

#---Funciones  Particulares ----------------



def g(a, x):
    a_1, a_2, a_3 = a
    g=a_1/((x - a_2) ** 2 + a_3)
    return g

def f_1(a, x, y, sigma):
    a_1, a_2, a_3 = a
    s = 0
    for i in range(len(x)):
        o = (y[i] - g(a, x[i])) / (((x[i] - a_2) ** 2 + a_3) * sigma[i] ** 2)
        s += o
    return s

def f_2(a, x, y, sigma):
    a_1, a_2, a_3 = a
    s = 0
    for i in range(len(x)):
        o = ((y[i] - g(a, x[i])) * ((x[i] - a_2))) / (((x[i] - a_2) ** 2 + a_3) ** 2 * sigma[i] ** 2)
        s += o
    return s

def f_3(a, x, y, sigma):
    a_1, a_2, a_3 = a
    s = 0
    for i in range(len(x)):
        o = ((y[i] - g(a, x[i]))) / (((x[i] - a_2) ** 2 + a_3) ** 2 * sigma[i] ** 2)
        s += o
    return s


f = [f_1, f_2, f_3]   #Vector de Funciones 

#---Funciones Newton Rapson Multidimencional 

da = [3.e-3,3.e-4,3.e-4] # MODIFICAR SI HAY MAS PARAMETROS  

print(da)
Nmax = 100 # Parametros 
Err0=5.e-12

def MapeoV(ta, x, y, sigma):
    F = []
    for func in f:
        F.append(func(ta, x, y, sigma))
    return np.array(F)  # Convertir a un array de numpy para operaciones vectoriales

# salto hacia adelante
def Mapeoplus(ta, i, j, da, x, y, sigma):
    tb = ta.copy() # Usar copy para no modificar ta original
    tb[j] += da[j]
    return f[i](tb, x, y, sigma)

# Funcion NEWTON RAPSON MULTIDMENCIONAL 
def NewtonRM(ta, da, Err0, Nmax, x, y, sigma):
    N = len(ta)  # Numero de paramteros de semilla 
    
    
    for it in range(Nmax + 1): 
        print('Iteracion numero =', it)
        F = MapeoV(ta, x, y, sigma)  
        print('Valor de F =', F)
        e0=F[0]
        e0=float(e0)
        e1=F[1]
        e1=float(e1)
        e2=F[2]
        e2=float(e2)
          
        # Verificar si la norma de F es menor que Err0
        if e0<Err0:
            if e1<Err0:
                if e2< Err0:
                    print("Condicion de error alcanzada. Terminando iteracion.")
                    break
                
#---------Preguntar POR que es que   la busqueda menor que no se cumplia correctamente aqui? 
        
        # Calculamos la jacobina  dfi/dxj
        df = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                df[i, j] = (Mapeoplus(ta, i, j, da, x, y, sigma) - f[i](ta, x, y, sigma)) / da[j] 
                #print(Mapeoplus(ta, i, j, da, x, y, sigma),'|',i,j )
                #print(f[i](ta, x, y, sigma),'|',i,j)
       
        #print(df)   #para observar el comportamiento del Jacobino 
        # Resolver el sistema lineal para encontrar la correccion
        try:
           df_inv = npm.inv(df)
        except np.linalg.LinAlgError:
           print("La matriz Jacobiana no es invertible. Terminando iteracion.")
           break
       
        
        # Resolver el sistema lineal para encontrar la correccion
        delta = np.dot(df_inv, -F)
        
        # Actualizar ta
        ta = ta + delta

    return ta 



#El algoritmo de newton es preciso , pero nesesita  un valor inicial cercano a la raiz para ser realmente efectivo , por eso usaremos en conjunto biseccion  y newton rapson 

ta=NewtonRM(ta, da, Err0, Nmax, x, y, sigma)
print('Entonces , tenemos que :\n')  


print('fr=',ta[0],'\n')
print('Er=',ta[1],'\n')
print('Gamma=',math.sqrt(4*ta[2]),'\n') 


 


x_test=np.linspace(min(x),max(x),n)
y_test=g(ta,x_test) 



# Grafica
# Configuracion para utilizar LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # tamano de la figura y calidad
# Datos originales
#ax.scatter(x, y, color='red', marker='o', )
# Plot de la interpolacion de Lagrange
ax.plot(x_test, y_test, '-', color='blue', label='Splines Cubicos',markersize=3, linewidth=2)
plt.errorbar(x, y, yerr=sigma, fmt='or', capsize=5,label='Datos')


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
