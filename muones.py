import numpy as np
import matplotlib.pyplot as plt 
import random 
# Constantes
m_pi = 139.6  # MeV/c^2
tau = 3.6e-8  # s
c = 3e8  # m/s
E_k = 200  # MeV
d = 20  # m
N0 = 1e6  # Numero inicial de piones

# ----Piones Monoenergeticos-----------

E_tot = E_k + m_pi  # Energia total en MeV
gamma = E_tot / m_pi
v = c * np.sqrt(1 - 1/gamma**2)
tau_lab = gamma * tau

# Distancia a recorrer y tiempo
t = d / v

# Probabilidad de supervivencia
P_surv = np.exp(-t / tau_lab)
N_surv1 = N0 * P_surv

print(f"Probabilidad de supervivencia: {P_surv:.6f}")
print(f"Numero de piones monoenergeticos  que sobreviven: {int(N_surv1)}") 


# ----Piones NO Monoenergeticos-----------

# Parametros de la distribucion gaussiana
E_mean = 200  # MeV
E_sigma = 50  # MeV

# Generar las energias cineticas
np.random.seed(425)  # Para reproducibilidad
E_kinetic = np.random.normal(E_mean, E_sigma, int(N0))

# Filtrar energias no fisicas (negativas)
E_kinetic = E_kinetic[E_kinetic > 0]

# Calcular el numero de piones que sobreviven
N_surv = 0
for E_k in E_kinetic:
    E_tot = E_k + m_pi
    gamma = E_tot / m_pi
    v = c * np.sqrt(1 - 1/gamma**2)
    tau_lab = gamma * tau
    t = d / v
    P_surv = np.exp(-t / tau_lab)
    if np.random.rand() < P_surv:
        N_surv += 1

print(f"Numero de piones no monoenergeticos  que sobreviven: {N_surv}") 

print('el error relativo es entonces:',(N_surv1-N_surv)/N_surv1)