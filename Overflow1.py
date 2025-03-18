
"""
@author: D.A. Vazquez Gutierrez
"""
N=20000
overflow=1
for n in range(1, N): #queremos que inicie en 1 la cuenta .
    overflow*=2
    print("|%2d"%n,"\t\t","|%2.5e"%overflow)
    
    