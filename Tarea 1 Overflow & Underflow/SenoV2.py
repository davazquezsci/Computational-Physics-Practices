"""
@author: D.A. Vazquez Gutierrez
"""
import math
import pandas as pd

pi = math.pi

# Definimos nuestra nueva funcion seno para todo valor de x 
def seno(x):
    r = x % (2 * pi)
    a = r
    suma = a
    sumamenos = 1
    amenos = 0
    n = 1
    while abs(a / sumamenos) > (10**(-16)):    #tamano del error absoluto 
        amenos = a
        sumamenos = suma
        n = n + 1
        a = (((-1) * r**2) * (amenos)) / (((2 * n) - 2) * ((2 * n) - 1))
        suma = suma + a
    return suma

#Creamos entonces una tabla para  poder encontrar el error relativo  entre senos . 

h = 15     #largo del arreglo 
particion = [0] * h
sumaPar = [0] * h
senoOr = [0] * h
error = [0] * h

for i in range(h):
    particion[i] = (120*(1+i)) / h #Rango de la particion 
    sumaPar[i] = seno(particion[i])
    senoOr[i] = math.sin(particion[i])
    error[i] = ((sumaPar[i] - senoOr[i]) / senoOr[i])

# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame({
    'Particion': particion,
    'Seno Aproximado': sumaPar,
    'Seno Original': senoOr,
    'Error Relativo': error
})


df_resultados.to_excel('resultados_seno4.xlsx', index=False) #Paso de listas de python a excel para despue pasarlas a tablas en LateX


print(df_resultados)
