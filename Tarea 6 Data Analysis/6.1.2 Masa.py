import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter 
from scipy.interpolate import CubicSpline

# Cargar los datos
data = pd.read_csv('Jpsimumu_Run2011A.csv') 
 

# Calcular la masa invariante (ya que tenemos la funcion calcular_masa_invariante)
def calcular_masa_invariante(row):
    E1, px1, py1, pz1 = row['E1'], row['px1'], row['py1'], row['pz1']
    E2, px2, py2, pz2 = row['E2'], row['px2'], row['py2'], row['pz2']
    masa_invariante = np.sqrt((E1 + E2)**2 - ((px1 + px2)**2 + (py1 + py2)**2 + (pz1 + pz2)**2))
    return masa_invariante

data['masa_invariante'] = data.apply(calcular_masa_invariante, axis=1)

#---Encontrar picos  de la grafica------

max_masa_invariante = data['masa_invariante'].max()
min_masa_invariante = data['masa_invariante'].min()

# Histograma de las masas invariantes
counts, bin_edges = np.histogram(data['masa_invariante'], bins=120, range=(min_masa_invariante , max_masa_invariante)) 
# Encuentra la parte central "el valor que supondremos " de cada "bin"
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


#Creacion de curva simulada por cubic Slines
cs=CubicSpline(bin_centers, counts)
N=10000 
x_test=np.linspace(min_masa_invariante, max_masa_invariante,N)
y_test=cs(x_test) 

#Ahora queremos escoger como altura minima para los picos , el pico mas pequeno previo a el pico maximo de todo el histograma

# Encontrar los indices de los picos
indices_picos = find_peaks(y_test, height=0)[0]  #La funcion de pandas , find_peaks, encuentra automaticamente los picos 
print(indices_picos)

# Encontrar el indice del bin con el maximo numero de repeticiones
indice_bin_max_repeticiones = np.argmax(y_test) 
print(indice_bin_max_repeticiones ,y_test[indice_bin_max_repeticiones],x_test[indice_bin_max_repeticiones])

# Filtrar los picos previos al bin con el maximo numero de repeticiones
indices_picos_previos = indices_picos[indices_picos < indice_bin_max_repeticiones]

# Encontrar el valor minimo de repeticiones entre los picos previos
valor_min_repeticiones_previos = np.min(y_test[indices_picos_previos])


# Detectar picos 
peaks, _ = find_peaks(y_test, height=valor_min_repeticiones_previos)    




# Graficar el histograma y los picos detectados
plt.figure(figsize=(7, 5), dpi=300)
plt.hist(data['masa_invariante'], bins=120, range=(min_masa_invariante , max_masa_invariante), color='skyblue', edgecolor='black', alpha=0.7)
#plt.plot(x_test,y_test, color='red', lw=2)
plt.plot(x_test[peaks], y_test[peaks], "x", color='black')

# Anadir etiquetas y titulo en formato LaTeX
plt.xlabel(r'Masa Invariante (GeV/c$^2$)', fontsize=14, color='black')
plt.ylabel(r'Número de Eventos', fontsize=14, color='black')
plt.title(r'                                     ', fontsize=16, color='black')

# Anadir una cuadrícula
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Ajustar margenes
plt.tight_layout()

# Guardar la figura en formato PNG con alta calidad
plt.savefig('masa_invariante_dimuones_peaks.png', format='png', dpi=300)

# Mostrar la figura
plt.show()

# Imprimir las posiciones de los picos
print("Picos detectados en bin centers:") 
print('\n  Masa  |' 'Numero de eventos ')
for i in range(len(peaks)):
    print('\n', x_test[peaks][i],'|',y_test[peaks][i])
