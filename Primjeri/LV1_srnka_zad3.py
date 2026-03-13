kraj = False
brojac = 0
lista = []
while kraj == False:
    brojac+=1
    try:
        print("Upisi broj:")
        lista.append(int(input()))
        print("Zelite li unijeti jos brojeva? (da/ne)")
        odgovor = input()
        if odgovor == "ne":
            kraj = True
    except:
        print("Niste unijeli broj, pokusajte ponovo.")

print("------------------------------")
print("Unijeli ste sljedece brojeve:")
for i in lista:
    print(i)
print("------------------------------")
print("Srednja vriejdnost:")
print(sum(lista)/brojac)
print("------------------------------")
print("Najveci broj:")
print(max(lista))
print("------------------------------")
print("Najmanji broj:")
print(min(lista))
print("------------------------------")

print(sorted(lista))