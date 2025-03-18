import numpy as np
import matplotlib.pyplot as plt

N = 3000 # Numero de PARTICULAS global 
t = 10000 # TIEMPO , mayor que el numero de particulas
nS = 10 #numero de simulaciones

#------Definicion de Funciones--------
def SIM(N,t):
    
    ni = N # Numero de particulas inicial del lado izquierdo igual a N
    ni_t = [ni] #En esta lista colocamos el numero de particulas del lado izquierdo segun vaya transcurriendo el tiempo t
    nd=0 #Numero de particulas inicial del lado derecho igual a N
    nd_t=[nd]  #En esta lista colocamos el numero de particulas del lado derecho  segun vaya transcurriendo el tiempo t
    
    for i in range(t): 
        s=np.random.rand() #Variable aleatoria si una particula sale de la izquierda 
        u=np.random.rand() #Variable aleatoria si una particula sale de la iderecha 
        k=np.random.rand() #Variable aleatoria que determina que lado de la caja tendra prioridad  para expulsar una particula 
        if (ni/N)>k:  #Prioridad en la izquierda 
            if ni>0:  #Nos aseguramos que la caja tenga particulas 
                if (ni/N)>s: #Conseguimos un numero aleatorio entre 0 y 1 y lo comparamos con la probabilidad ni/N
                    ni -= 1 # En el caso de ser cierto , una particula va a la derecha , perdiendo una del lado izquierdo.
                    nd += 1 
                elif nd>0 :       
                    if (nd/N)>u:
                        ni += 1 # En el caso de ser cierto , una particula va a la derecha , perdiendo una del lado izquierdo.
                        nd -= 1
        else:#Prioridad en la iderecha 
            if nd>0: #Nos aseguramos que la caja tenga particulas 
                if (nd/N)>s: #Conseguimos un numero aleatorio entre 0 y 1 y lo comparamos con la probabilidad ni/N
                    nd -= 1 # En el caso de ser cierto , una particula va a la derecha , perdiendo una del lado izquierdo.
                    ni += 1 
                elif ni>0 :       
                    if (ni/N)>u:
                        nd += 1 # En el caso de ser cierto , una particula va a la derecha , perdiendo una del lado izquierdo.
                        ni-= 1
        ni_t.append(ni) #Anadimos el nuevo numero de particulas en el lado izquierdo 
    print(nd)
    return ni_t

simulaciones  = [] #Guardamos simulaciones
for i in range(nS):
    simulaciones.append(SIM(N, t)) # Hacemos un nS veces la simulacion


time = np.arange(t) #arreglo en el tiempo para la solucion analitica
sol_an = (N/2) * (1 + np.exp(-2 * time / N)) #calculo analItico
ProM = np.mean(simulaciones, axis=0) # Promedio de todas las simulaciones // Para que se haga esto sobre las columnas , se coloca axis=0 

#-----GRAFICACION ---------------
# Definimos el tamano de la figura
plt.figure(figsize=(5, 5))

# Ajustes de fuente a Courier New
plt.rc('font', family='Courier New')


plt.figure(1) 
for i in simulaciones:
    plt.plot(i, alpha=0.4) #Alfa determina lo translucido de la grafica
    
plt.plot(ProM,color="black",linewidth=1.5, label='Promedio')
plt.plot(time, sol_an, 'r', label='Solucion analitica')  
    
plt.xlabel("Tiempo [s]")
plt.ylabel("Numero de particulas en el lado izquierdo (ni)")
plt.title("PARTICULAS EN UNA CAJA // CORRECCION ")
plt.legend()
plt.grid(True)
plt.show()
    