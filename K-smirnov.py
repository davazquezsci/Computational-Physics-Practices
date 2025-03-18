import numpy as np
from scipy.stats import kstest
import random
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Generar una muestra de datos utilizando la funcion random()
n = 200# Tamano de la muestra
N = 1000
r=20
w=n//r
alpha=0.1

data = [random.random() for _ in range(n)]

dataS = sorted(data)


#Distribucion Uniforme
def F(x): 
    h = uniform.cdf(x, loc=0, scale=1)
    return h

#Definicion de funciones  de Kolmogorov -Smirnov 

def kMENOS(datos, distribucion): 
    num = len(datos)
    q = np.zeros(num) 
    for j in range(num):
        q[j] = distribucion(datos[j]) - (j / num)
    return  np.sqrt(num)*np.max(q) 

def kMAS(datos, distribucion): 
    num = len(datos)
    q = np.zeros(num) 
    for j in range(num):
        q[j] = (j + 1) / num - distribucion(datos[j])
    return np.sqrt(num)*np.max(q) 


def Local(datos,Cmenos,Cmas,distribucion,r):
    num=len(datos) 
    KSmas=np.zeros(r)
    KSmenos=np.zeros(r)
    e=num//r #Dividimos el rango de nuestra muestra en pequenas partes 
    
    dat=np.zeros(e)
    for i in range(r):
        for j in range(e):
            dat[j]=datos[j+i]
        datS=sorted(dat)
        KSmas[i]=Cmas(datS,distribucion)
        KSmenos[i]=Cmenos(datS,distribucion)     
        
    Kmas=Cmas(sorted(KSmas),F)
    Kmenos=Cmenos(sorted(KSmenos),F)
    return KSmas,KSmenos,Kmas,Kmenos
        
    
def D(n,a):# valor critico
    return np.sqrt(-(1/(2))*np.log(a/2))


# -----Aplicar la prueba de Kolmogorov-Smirnov GLOBAL ARTESANAL ---
kmas=kMAS(dataS,F)
kmenos=kMENOS(dataS,F)

print('----Estadistico de prueba KS Artesanal SOLO GLOBAL----')

print(f'K+{n}:',kmas)
print(f'K-{n}:',kmenos)  

Dn1=max(kmas,kmenos)  

print(f'D{n}:',Dn1) 
print(f'D{alpha}:',D(n,alpha)) 

if Dn1<=D(n,alpha):
    print('Se acepta la hipotesis de que la funcion random crea una dsitribucion uniforme ')
else:
    print('Se descarta la hipotesis de que la funcion random crea una dsitribucion uniforme ')
    

# -----Aplicar la prueba de Kolmogorov-Smirnov Parche LOCAL ARTESANAL ---
KSmas,KSmenos,Kmas,Kmenos=Local(data,kMENOS,kMAS,F,r)

print('----Estadistico de prueba KS Artesanal parche LOCAL----')

print(f'K+{w}:',Kmas)
print(f'K-{w}:',Kmenos) 

Dn2=max(Kmas,Kmenos)  

print(f'D{w}:',Dn2) 
print(f'D{alpha}:',D(w,alpha)) 

if Dn2<=D(w,alpha):
    print('Se acepta la hipotesis de que la funcion random crea una dsitribucion uniforme ')
else:
    print('Se descarta la hipotesis de que la funcion random crea una dsitribucion uniforme ')

#-----GRAFICACION ----------
# Definimos el tamano de la figura
plt.figure(figsize=(12, 6))

# Ajustes de fuente a Courier New
plt.rc('font', family='Courier New')

y_ecdf = np.arange(1, n + 1) / n
x_ecdf = np.sort(data)

# Calcular la funcion de distribucion acumulada para KSmas y KSmenos
y_ksmas = np.arange(1, r + 1) / r

# Crear figura y subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Graficar la ECDF de los datos en el primer subplot
ax1.step(x_ecdf, y_ecdf, where='post', label='ECDF')
ax1.plot(np.linspace(0, 1, N), F(np.linspace(0, 1, N)), label='CDF Uniforme')
ax1.set_xlabel('x')
ax1.set_ylabel('F(x)')
ax1.set_title('ECDF y CDF Uniforme')
ax1.grid(True)
ax1.legend()

# Graficar la distribucion acumulada de KSmas en el segundo subplot
ax2.step(np.sort(KSmas), y_ksmas, where='post', label='KSmas')
ax2.plot(np.linspace(min(KSmas), max(KSmas), N), uniform.cdf(np.linspace(min(KSmas), max(KSmas), N),loc=min(KSmas), scale=max(KSmas)-min(KSmas)), label='CDF Uniforme')
ax2.set_xlabel(f'KSmas{w}')
ax2.set_ylabel('Probabilidad acumulada')
ax2.set_title('Distribucion acumulada de KSmas')
ax2.grid(True)
ax2.legend()


plt.tight_layout()
plt.show()
