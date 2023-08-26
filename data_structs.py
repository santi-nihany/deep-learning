# list
lista = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lista2 = [44, 12, 29]

print("lista = ", lista)
print("lista2 =", lista2, "\n")

print("lista[-2]: ", lista[-2])
print("lista[0:3]: ", lista[0:3])
print("lista[0:9:2]: ", lista[0:9:2])
print("lista * 2: ", lista * 2)
print("len(lista): ", len(lista))
print("4 in lista: ", 4 in lista)
print("lista + lista2 = ", lista + lista2)


# tuple
tupla = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

print("\ntupla = ", tupla)
print("tupla[1]: ", tupla[1])

# dictionary
diccionario = {
    "nombre": "Carlos",
    "edad": 22,
    "cursos": ["Python", "Django", "JavaScript"],
}
print("\ndiccionario = ", diccionario)
print("diccionario['nombre']: ", diccionario["nombre"])
print("diccionario['edad']: ", diccionario["edad"])


# set
conjunto = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

print("\nconjunto = ", conjunto)
print("len(conjunto): ", len(conjunto))
conjunto.add(11)
print("conjunto.add(11) = ", conjunto)
conjunto.remove(11)
print("conjunto.remove(11) = ", conjunto)
conjunto.clear()
print("conjunto.clear() = ", conjunto)
conjunto.update([1, 2, 3, 4])
print("conjunto.update([1, 2, 3, 4]) = ", conjunto)
