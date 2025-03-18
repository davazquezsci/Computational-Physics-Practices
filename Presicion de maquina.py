
# Determina aproximadamente la precision de maquina

paso=1.0  
pasomas=0   #tiene el valor de 1 mas paso dividido en factores de 2. 


while 1!=pasomas:  #Mientras estos sean diferentes, el ciclo se repitira indefinidamente. 
    paso=paso/2      #reducimos en factor de 2 a paso 
    pasomas=1+paso   #asignamos un valor a paso mas en funcion de paso 
    
print('Presicion de maquina:',"%e"%paso,'valor del cambio',pasomas)

print('Podemos ver que la maquina no puede diferenciar entre el 1 y ',pasomas,'el cual matematicamente es diferente de 1 , pero numericamente no.')