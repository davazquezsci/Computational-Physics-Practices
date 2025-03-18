import pandas as pd #Biblioteca que se enfoca en la manipulacion de datos , principalmente los estructurados . 
import numpy as np

# Cargar los datos
data = pd.read_csv('Jpsimumu_Run2011A.csv')  #comando con el cual cargamos los datos desde el archivo csv

# Definir una funcion para calcular la masa invariante
def calcular_masa_invariante(row):
    # Energias de los muones
    E1 = row['E1']
    E2 = row['E2']
    
    # Componentes del momento de los muones
    px1 = row['px1']
    py1 = row['py1']
    pz1 = row['pz1']
    px2 = row['px2']
    py2 = row['py2']
    pz2 = row['pz2']
    
    # Energia total del sistema
    E_total = E1 + E2
    
    # Componentes del momento total del sistema
    px_total = px1 + px2
    py_total = py1 + py2
    pz_total = pz1 + pz2
    
    # Cuadrado del momento total
    p_total_squared = px_total**2 + py_total**2 + pz_total**2
    
    # Masa invariante del sistema
    masa_invariante = np.sqrt(E_total**2 - p_total_squared)
    return masa_invariante

# Aplicar la funcion a cada fila del DataFrame
data['masa_invariante'] = data.apply(calcular_masa_invariante, axis=1)   # Anade una nueva columna al DataFrame ||=  Aplica una funcion a cada fila del DataFrame.

# Encontrar el valor maximo y el minimo  de las masas invariantes.
max_masa_invariante = data['masa_invariante'].max()
min_masa_invariante = data['masa_invariante'].min()


print(data.head())  # Mostrar las primeras filas con la masa invariante calculada


#------Graficacion -------------------

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Listar estilos disponibles
print(plt.style.available)

# Usar un estilo disponible
plt.style.use('ggplot')

# Configurar matplotlib para usar LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)  # Tamano de la figura y calidad

# Histograma de las masas invariantes
ax.hist(data['masa_invariante'], bins=120, range=(min_masa_invariante, max_masa_invariante), color='skyblue', edgecolor='black')

# Anadir etiquetas y titulo en negro intenso
ax.set_xlabel(r'Masa Invariante (GeV/c$^2$)', fontsize=14, color='black')
ax.set_ylabel(r'Numero de Eventos', fontsize=14, color='black')
ax.set_title(r'                                  ', fontsize=16, color='black')

# Configurar las etiquetas de los ejes
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Anadir una cuadricula
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Ajustar las etiquetas de los ejes en negro intenso
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')

# Ajustar margenes
plt.tight_layout()

# Guardar la figura en formato PNG con alta calidad
plt.savefig('masa_invariante_dimuones.png', format='png', dpi=300)

# Mostrar la figura
plt.show()



