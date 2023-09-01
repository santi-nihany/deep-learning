
VALORES = [-0.12, -0.21, 0.29]
DUREZA = [60, 30, 60, 30, 0, 60, 0, 0, 30]
TEMPERATURA = [-12, 20, 0, 7, 11, -30, 63, 94, 45]
func = []

for i in range(9):
    func.append(VALORES[0] * TEMPERATURA[i] +
                VALORES[2] * DUREZA[i] + VALORES[1])
    print(VALORES[0] * TEMPERATURA[i] + VALORES[2] * DUREZA[i] + VALORES[1])
