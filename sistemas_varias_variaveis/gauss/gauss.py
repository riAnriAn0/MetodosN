def gauss_seidel(a, b, epsilon=0.0001, max_iter=1000):
    n = len(a)
    x = [0.0 for _ in range(n)]

    for iteration in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            sum_ax = sum(a[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_ax) / a[i][i]

        # Verifica o critério de parada
        if all(abs(x_new[i] - x[i]) < epsilon for i in range(n)):
            print(f"Convergiu em {iteration+1} iterações.")
            return x_new

        x = x_new

    raise Exception("O método não convergiu no número máximo de iterações.")

a = [
    [1.0234, -2.4567,1.2345],
    [5.0831,1.2500,0.9878],
    [-3.4598,2.5122,-1.2121]
]

b = [6.6728,6.5263,-11.2784]

solucao = gauss_seidel(a, b)
print("Solução:", solucao)
