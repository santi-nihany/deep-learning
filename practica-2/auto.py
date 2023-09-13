import pandas as pd
df = pd.read_csv("./dataFiles/automobile-simple.csv")

print(df[["engine-size", "highway-mpg"]].corr())

ite = 0
atipicos = 0
for noms in df.columns[4:]:

    quartiles = df[noms].quantile([0.25, 0.5, 0.75]).values
    RIC = quartiles[2]-quartiles[0]
    ok = True
    ite = 0
    # print(noms)
    # print(df[noms])
    while ((ok) and (ite < len(df[noms]))):

        if (df[noms][ite] > (quartiles[2] + 1.5*RIC)) or (df[noms][ite] < (quartiles[0] - 1.5*RIC)):
            ok = False
            atipicos = atipicos+1
            # print("Limites : ",(quartiles[2] + 1.5*RIC),"-",(quartiles[0] - 1.5*RIC)," --- Valor atipico: ",df[noms][ite])
        ite = ite+1

print("La cantidad de valores atipicos es: ", atipicos)

# print(df)
