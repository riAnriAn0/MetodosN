def arredondar(numero, casas=0):
    fator = 10 ** casas
    valor = numero * fator
    parte_inteira = int(valor)
    parte_decimal = valor - parte_inteira

    if parte_decimal >= 0.5:
        parte_inteira += 1

    return parte_inteira / fator


# Entrada de dados
ERRO = 0.0001
MAX_ITERACOES = 1000
matriz = [ [1.427, -3.948, 10.383, -32.793],
          [ -2.084, 6.425, -0.083, 36.672],
 [15.459 ,-2.495 ,-1.412, -6.557]
]
NUMERO_DE_VARIAVEIS = 3# int(input("NUMERO DE VARIAVEIS DA EQUAÇÃO: "))
LINHAS = NUMERO_DE_VARIAVEIS
COLUNAS = NUMERO_DE_VARIAVEIS + 1

# print("Digite os coeficientes das equações (inclusive o termo independente):")
# for l in range(LINHAS):
#     while True:
#         entrada = list(map(float, input(f"EQUAÇÃO [{l+1}]: ").split()))
#         if len(entrada) != COLUNAS:
#             print(f"Erro: Insira {COLUNAS} valores (coeficientes + termo independente).")
#         else:
#             matriz.append(entrada)
#             break

s1 = [0] * NUMERO_DE_VARIAVEIS
execucoes = 0
run = True

while run and execucoes < MAX_ITERACOES:
    s2 = []
    vet_erro = []
    
    for i in range(NUMERO_DE_VARIAVEIS):
        soma = 0.0
        for j in range(NUMERO_DE_VARIAVEIS):
            if j != i:
                soma += matriz[i][j] * s1[j]
        
        if matriz[i][i] == 0:
            raise ZeroDivisionError(f"Divisão por zero detectada na linha {i+1}. Verifique a matriz.")

        novo_xi = (1 / matriz[i][i]) * (matriz[i][NUMERO_DE_VARIAVEIS] - soma)
        s2.append(arredondar(novo_xi, 4))

    for i in range(NUMERO_DE_VARIAVEIS):
        vet_erro.append(abs(s2[i] - s1[i]))

    run = any(erro > ERRO for erro in vet_erro)
    s1 = s2.copy()
    execucoes += 1

print("\nMATRIZ AUMENTADA\n")
for linha in matriz:
    print("[", " ".join(f"{num: .4f}" for num in linha), "]")

print("\nVETOR SOLUÇÃO\n")
for i in range(NUMERO_DE_VARIAVEIS):
    print(f"X{i+1} : {s2[i]}")

print(f"\nERRO: {vet_erro}")
print(f"Numero de execuções: {execucoes}\n")

if execucoes == MAX_ITERACOES:
    print("Aviso: Número máximo de iterações atingido. A solução pode não ter convergido.")
