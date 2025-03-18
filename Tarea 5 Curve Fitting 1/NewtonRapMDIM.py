
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
f = data.iloc[1, 1:].astype(float).values
sigma = data.iloc[2, 1:].astype(float).values 
sigma=sigma

# Definicion de  funciones f que minimizan \chi^2   --------------

#data.iloc[1, 1:] selecciona la segunda fila (indice 1) excluyendo el primer valor (que es el encabezado de la fila). Luego, se convierte a tipo float y se extrae como un array de valores.
x=E
y=f

error=0.001

N=3  #dimencion  del problema a resolvedr  
n=100 #Numero de espacios  de los parametros 

a_min = [10, 10, 10]  # Maximos y minimos de cada  dimencion  de a_i , CAMBIAR SI AUMENTAN LOS PARAMETOS N !!!
a_max = [100, 100, 1000]  #ojo con los parametros de a_n , ya que hay que evitar indeterminaciones en las matrices 

a = []

for i in range(N):
    a.append(np.linspace(a_min[i], a_max[i], n))

a = np.meshgrid(*a)   #generalizacion para cualquier n de a[0],a[1],a[2]= np.meshgrid(a[0],a[1],a[2])


def g(a,x): 
    a_1, a_2, a_3 = a  #modificacion para usar la meshgrid , pero solo funcionara  con matrices  N x n 
    f=a_1/((x-a_2)**2+a_3)
    return f

def f_1(a,x,y,sigma):
    a_1, a_2, a_3 = a     #CAMBIAR SI AUMENTAN LOS PARAMETOS N !!!
    s=0
    for i in range(len(x)):
        o=(y[i]-g(a,x[i]))/(((x[i]-a_2)**2+a_3)*sigma[i]**2)
        s=s+o
    return(s)  

def f_2(a,x,y,sigma):
    a_1, a_2, a_3 = a     #CAMBIAR SI AUMENTAN LOS PARAMETOS N !!!
    s=0
    for i in range(len(x)):
        o=((y[i]-g(a,x[i]))*(-2*a_1*(x[i]-a_2)))/(((x[i]-a_2)**2+a_3)**2*sigma[i]**2)
        s=s+o
    return(s)  

def f_3(a,x,y,sigma):
    a_1, a_2, a_3 = a     #CAMBIAR SI AUMENTAN LOS PARAMETOS N !!!
    s=0
    for i in range(len(x)):
        o=((y[i]-g(a,x[i]))*(-a_1))/(((x[i]-a_2)**2+a_3)**2*sigma[i]**2)
        s=s+o
    return(s)  

f= [f_1, f_2, f_3]   #definimos un "vector " de funciones 

#r=[40,40,70]

#print(f_3(r,x,y,sigma))


#-----Metodos Foward Diference---------

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
Ra = []

for i in range(N):
    Ra.append(a_max[i] - a_min[i])

ta = [90,45,600]
ta=np.array(ta) 


'''
for i in range(N):
    ta.append(Ra[i]/2)
'''

da=3.e-3
err = 0.001
Nmax = 100 # Parametros 
Err0=1.e-7
Err01=0.015  

ta[1]+=da  
ta[1] += da



def MapeoV(ta, x, y, sigma):
    F=[]
    for i, func in enumerate(f):
        F.append(func(ta, x, y, sigma))
        '''
        Aqui, func es la funcion actual que estamos evaluando en la iteracion actual del bucle. Llamamos a func con los valores de entrada
        '''
    return np.array(F)  

def Mapeoplus(ta, i,j, da, x, y, sigma):
    tb=ta
    tb[j]+=da 
    return f[i](tb,x,y,sigma)
#esta funcion solo es para hacer el salto hacia adelante  fi(xj+dxj)

    

def NewtonRM(ta, da, Err0, Nmax):
    df=np.zeros((N,N))
    
    for it in range(0, Nmax + 1): 
        print('Iteracion numero =',it)
        F=MapeoV(ta, x, y, sigma)  
        print('valor de F=',F)
        
         #para dim 3 < , hay que modificar esta parte 
        if abs(F[0])<Err0 and abs(F[1])<Err0 and abs(F[2])<Err0 :
            print("Condición de error alcanzada. Terminando iteración.")
            break
        '''
        if np.linalg.norm(F)<Err0:
           print("Condición de error alcanzada. Terminando iteración.")
           break
    
        elif F is None:
            print('\n Valor de funcion es None en x=', ta)
            break
        '''
       # print('Iteracion=', it, 'x=', t0, 'f(x)=', F) 
        for i in range(N):
           for j in range(N): 
               df[i,j] = Mapeoplus(ta,i,j,da,x,y,sigma) - f[i](ta,x,y,sigma) / da  # Central difference  
        try:
             delta = np.linalg.solve(df, -F)
        except np.linalg.LinAlgError:
             print("Error al resolver el sistema lineal.")
             break
        ta = ta + delta
    return ta 
'''
       # if df == 0:
            # Division por cero en df. No se puede continuar.
           # break
        da = -npm.inv(df)*F     #encontramos el inverso de la matris de dfi/dxj  y encontramos el salto delta x 
        ta =ta+da  # Nueva propuesta
   '''
        
#El algoritmo de newton es preciso , pero nesesita  un valor inicial cercano a la raiz para ser realmente efectivo , por eso usaremos en conjunto biseccion  y newton rapson 

ta=NewtonRM(ta, da, Err0, Nmax)
print('Entonces , tenemos que :\n') 

print('fr=',ta[0],'\n')
print('Er=',ta[1],'\n')
print('Gamma=',math.sqrt(4*ta[2]),'\n') 

Tx=[30,78,756.3] 

print(MapeoV(Tx,x,y,sigma)) 
print(f_1(Tx,x,y,sigma))


