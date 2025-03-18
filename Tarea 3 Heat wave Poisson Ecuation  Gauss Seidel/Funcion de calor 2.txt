import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import math

L = 1  # longitud de la barra 
#alfa = 9.786e-5 #Coeiciente de difusividad termica  aluminio 
alfa = 8.091e-8
To = 100  # temperatura inicial , 100 grados celius 
tf=6000#tiempo final , en este caso suponemos que transcurren 2 minutos 
a = np.sqrt(alfa) 
Nx = 1000 # Numero de particiones del eje x 
Nt = 1000 # numero de particiones  del eje y 

# Parametros
eta = alfa*(tf/Nt)/((L/Nx)**2) # Parametro eta , en funcion de la ecuacion 2 
print('Eta es igual a :',eta)

# Inicializacion de la malla phi
phi = np.zeros((Nx, Nt))

# Condiciones iniciales
phi[:, 0] = To  # Todas las posiciones tienen la temperatura inicial

# el : en el phi[:,0] indica que sin importar el numero  en la coordenada x, para cualquiera que este relacionada con y=0 
#tomara el valor que se quiera , en este caso To (temperatura incial)

# Condiciones de contorno
phi[0, :] = 0
phi[-1, :] = 0

#el -1 en el phi[-1, :] , es una forma en la que sin conocer la longitud de la dimencion en x de la malla , podemos indicar
#el ULTIMO elmento , de igual manera se puede indicar el PENULTIMO elemento empleando el -2 y asi sucesivamente. 
#en este caso decimos que para todo ultimo valor de x en la malla, este simepre valdra 0 


# Iteracion de la EDP
for j in range(1, Nt): #empezamos en 1 por que la ilera de t=0 ya esta inicializada 
    for i in range(1, Nx-1): #vamos de 1 a Nx-1 por que tanto el primero (0) como el ultimo ya estan incializados en x 
        phi[i, j] = phi[i, j-1] + eta * (phi[i+1, j-1] + phi[i-1, j-1] - 2 * phi[i, j-1]) #Por como esta definido phi en la ecuacion 9 , hacemos un cambio de j+1 a j  

print()


# Crear una cuadricula 2D
x = np.linspace(-L/2, L/2, Nx)
t = np.linspace(0, tf, Nt)
t, x = np.meshgrid(t, x)  # Aqui t se transpone con x

# Grafico 3D de la superficie
fig = plt.figure(figsize=(12, 12),dpi=300) #dpi=300 aumenta la calidad de  la grafica , y la proporcion de figsize hace que los labels sean de un tamano apropiado 
ax = fig.add_subplot(111, projection='3d') #el subpot 111, se refiere a que se proyecta la  grafica 1, de un total de graficas 1x1 , hay mas configuraciones 212, 223, etc 
surf=ax.plot_surface(x, t, phi, cmap='viridis', linewidth=0, antialiased=False)  
# Anadir etiquetas y titulo
ax.set_xlabel('Distancia(m)', fontsize=12)
ax.set_ylabel('Tiempo (s) ', fontsize=12)
ax.set_zlabel('Temperatura(C)', fontsize=12)
ax.set_title('                                          ', fontsize=50) 

# Ajustes para mejorar la visualizacion
ax.view_init(elev=30, azim=45)  # Cambiar la elevacion y el angulo de vision
fig.colorbar(surf, shrink=0.5, aspect=8)  # Anadir barra de color  


plt.show()