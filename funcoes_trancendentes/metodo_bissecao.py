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

import math

# Definir constante de tolerância
TOL = 1e-4 

def aprox(num, cdecimais):
    is_n = False
    if num < 0:
        is_n = True
        num *= -1

    fator = 1/cdecimais
    valor = num * fator
    parte_inteira = int(valor)
    part_decimal = valor - parte_inteira

    if part_decimal > 0.5:
        parte_inteira += 1

    if is_n:
       return -1 * (parte_inteira/fator)
    else:
        return parte_inteira/fator

def f(x):
    return x**3 - math.e**(2*x) + 3

def bissecao(f, a, b, TOL, N):
    i = 1
    fa = aprox(f(a),TOL)
    fb = aprox(f(b),TOL)

    # Verificar se os valores iniciais estão corretos
    if a >= b or a == b:
        print('\nExtremos do intervalo inválidos!')
        return None, None
    elif fa * fb > 0:
        print('\nValores iniciais não satisfazem o teorema de Bolzano!')
        return None, None

    while i <= N:
        # iteracao da bissecao
        p = aprox((a + (b - a) / 2),TOL)
        fp = aprox(f(p),TOL)
        # condicao de parada
        if (fp == 0) or ((b - a) / 2 < TOL):
            return p, i
        # bissecta o intervalo
        i = i + 1
        if fa * fp > 0:
            a = p
            fa = fp
        else:
            b = p
    raise NameError('Num. max. de itererações foi atingido!')

retorno = bissecao(f, 4, 6, TOL, 100)

print(f"Raiz aproximada: {retorno[0]}")
print(f"Numero de itereações: {retorno[1]}")
