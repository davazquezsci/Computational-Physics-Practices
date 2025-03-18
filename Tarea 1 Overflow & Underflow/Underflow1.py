
"""

@author: D.A. Vazquez Gutierrez

"""

underflow=1 
llimite=1
s=0
while underflow!=0:  #nuestra condicion de parada sera cuando el valor sea indistinguible del cero. 
    s+=1 
    limite=underflow  #guardamos el valor previo a esta iteracion,por si llegara a haber underflow .
    underflow/=2 
    
print("El valor de limite  de underflow es:","%e"%(limite), "con la iteracion","%2d"%(s-1),)




