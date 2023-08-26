import math

nombres = ["Juan Sebastián", "Juan Roman", "Rodrigo", "Juan Sebastián"]
apellidos = ["Verón", "Riquelme", "Braña"]

# 2
# Genere el código necesario para recorrer simultáneamente 2 listas con la misma cantidad de elementos e imprima
# los mismos utilizando un único for (tip: función zip).

for nombre, ape in zip(nombres, apellidos):
    print(nombre, ape)

# 3
# Implemente una función que a partir de la lista que recibe cómo parámetro, retorne una nueva lista sin elementos
# repetidos. Compruebe su correcto funcionamiento.


def noRepetidos(lista: list):
    listaSinRepetidos = []
    for elemento in lista:
        if elemento not in listaSinRepetidos:
            listaSinRepetidos.append(elemento)
    return listaSinRepetidos


print("\nlista = ", nombres)
print(noRepetidos(nombres))

# 4
# Implemente una función que calcule la distancia entre 2 puntos (2D). Utilice la función sqrt del paquete math para
# implementarla y compruebe el correcto funcionamiento de la misma


def distancia(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


print("distancia(1,1),(1,2): ", distancia(1, 1, 1, 2))
print("distancia(1,1),(2,2): ", distancia(1, 1, 2, 2))
