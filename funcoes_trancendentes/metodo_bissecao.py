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
    return 0.2*x**3 - 3.006*x**2 + 15.06*x - 25.15 

def tab(n, An, Bn, Xn, fxn, erro = 0.0):
    if n == 1:
        print(" N  |     An     |     Bn     |     Xn     |     fxn    |     Erro  ")
    
    print("+---------------------------------------------------------------")
    print(f"{n:>3} | {An:>10.6f} | {Bn:>10.6f} | {Xn:>10.6f} | {fxn:>10.6f} | {erro:>10.6f}")
    

def bissecao(f, a, b, TOL, N):
    i = 1
    fa = aprox(f(a),TOL)
    fb = aprox(f(b),TOL)
    erro = 0.0

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
        
        tab(i, a, b, p, fp, erro)
        erro = f(p) - fp

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

retorno = bissecao(f,5.01, 5.5 , TOL, 100)

print(f"\nRaiz aproximada: {retorno[0]}")
print(f"Numero de itereações: {retorno[1]}")
