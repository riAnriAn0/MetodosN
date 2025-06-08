#    Função Python ------------ Substituto JAX (jnp)
#        math.log(x)	        jnp.log(x)
#        math.exp(x)	        jnp.exp(x)
#        math.sin(x)	        jnp.sin(x)
#        math.cos(x)	        jnp.cos(x)
#        math.tan(x)	        jnp.tan(x)
#        math.sqrt(x)	        jnp.sqrt(x)
#        math.pow(x, y)	        jnp.power(x, y)
#        abs(x)		            jnp.abs(x)
#                               jnp.e() =~  2.7182817

import jax
import jax.numpy as jnp

# Definir constante de tolerância
TOL = 1e-4

# Definir função de aproximação
def aprox(num, cdecimais):
    is_n = False
    if num < 0:
        is_n = True
        num *= -1

    fator = 1 / cdecimais
    valor = num * fator
    parte_inteira = int(valor)
    part_decimal = valor - parte_inteira

    if part_decimal > 0.5:
        parte_inteira += 1

    if is_n:
        return -1 * (parte_inteira / fator)
    else:
        return parte_inteira / fator

# Definir função f
def f(x):
       return x**2 - 10*jnp.log(x) - 5

# Definir 2º derivada da função f
def f2(f, x):
    f1 = jax.grad(f)
    ff = jax.grad(f1)
    return aprox(ff(x), TOL)

def cordas(f, a, b, TOL, N):
    i = 1
    erro = 0.0
    fa = f(a)
    fb = f(b)

    # Verificar se os valores iniciais estão corretos
    if a >= b or a == b:
        print('\nExtremos do intervalo inválidos!')
        return None, None
    elif fa * fb > 0:
        print('\nO intervalo não contem uma raiz!')
        return None, None

    ffa = f2(f, a)
    ffb = f2(f, b)

    c = 0
    xn = 0

    # Definir o ponto fixo C
    if (fa * ffa > 0):
        c = a
        xn = b
    elif (fb * ffb > 0):
        c = b
        xn = a
    else:
        print('\nNão foi possivel determinar o ponto fixo!')
        return None, None

    fxn = f(xn)
    fc = f(c)

    while i <= N:
        xnn = aprox(xn - (fxn * (xn - c)) / (fxn - fc), TOL)
        fxnn = f(xnn)

        # condicao de parada
        if (fxnn == 0) or (jnp.abs(xnn - xn) < TOL):
            return aprox(xnn, TOL), i
        # Atualizar valores
        xn = xnn
        fxn = fxnn
        i += 1
    print('\nNumero max de interações atingido!')

retorno = cordas(f, 0.5, 1.0, TOL, 100)
print(f"Raiz aproximada: {retorno[0]}")
print(f"Número de iterações: {retorno[1]}")
