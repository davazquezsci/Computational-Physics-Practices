
"""

@author: D.A. Vazquez Gutierrez
"""

N=2000 #para un N lo suficientemente grande 

while True:
    try:
        overflow=1    #valor inicial del overflow 
        s=0          #Contamos los pasos 
        for n in range(N):
            
               overflow*=2
               s+=1
               
           
        print("El valor del limite de overflow es:","%e"%(overflow), "con la iteracion","%2d"%s,) #Si esto arroja un error de Overflow, lo manda al "except"
        break  
        
    except OverflowError:   #Si se llega a un overflow significa que el numero N que elegimos es demaciado grande 
        N=N-1                #Disminuimos N          
       
      
 
        
        