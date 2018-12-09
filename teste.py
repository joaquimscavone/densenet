arq = open('pesos/treino.txt', 'r')
texto = arq.readlines()
arq.close()
v=int(texto[0]);
v+=1
print(v)
