import scipy.special as ss

# Orden de la función de Bessel
n = 3

# Argumento de la función de Bessel
x = 1

# Calcular la función de Bessel de primera especie J_n(x)
bessel_value = ss.spherical_jn(n, x)

print(f"J_{n}({x}):", bessel_value)