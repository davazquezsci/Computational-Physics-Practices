import math

def bisection(f, a, b, tol, max_iter):
    """
    Implementación del método de bisección para encontrar una raíz de una función.

    Args:
        f: Función cuya raíz se busca.
        a: Extremo izquierdo del intervalo inicial.
        b: Extremo derecho del intervalo inicial.
        tol: Tolerancia para el error absoluto entre iteraciones consecutivas.
        max_iter: Número máximo de iteraciones permitidas.

    Returns:
        float: Aproximación de la raíz encontrada.
        int: Número de iteraciones realizadas.
    """
    # Verificar si la raíz está dentro del intervalo [a, b]
    if f(a) * f(b) >= 0:
        raise ValueError("La función no cambia de signo en el intervalo dado.")

    # Inicializar variables
    iter_count = 0
    while (b - a) / 2 > tol and iter_count < max_iter:
        c = (a + b) / 2  # Punto medio del intervalo
        if f(c) == 0:
            return c, iter_count
        elif f(c) * f(a) < 0:
            b = c  # La raíz está en el subintervalo [a, c]
        else:
            a = c  # La raíz está en el subintervalo [c, b]
        iter_count += 1

    return (a + b) / 2, iter_count

# Ejemplo de uso:
def f(x):
    # Aquí defines tu función
    return x**2 - 4  # Por ejemplo, para encontrar la raíz de x^2 - 4 = 0

# Parámetros del método de bisección
a = 0  # Extremo izquierdo del intervalo inicial
b = 3  # Extremo derecho del intervalo inicial
tol = 1e-6  # Tolerancia para el error absoluto entre iteraciones consecutivas
max_iter = 1000  # Número máximo de iteraciones permitidas

# Llamar al método de bisección
root, iterations = bisection(f, a, b, tol, max_iter)
print("Raíz aproximada:", root)
print("Iteraciones realizadas:", iterations)
