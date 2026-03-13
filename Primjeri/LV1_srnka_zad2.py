try:
    bodovi = float(input("Postotak bodova između 1 i 0: "))
    if bodovi < 0 or bodovi > 1:
        print("Neispravan unos")
    if bodovi >= 0.9 and bodovi <= 1:
        print("Ocjena: 5")
    elif bodovi >= 0.8 and bodovi <= 1:
        print("Ocjena: 4") 
    elif bodovi >= 0.7 and bodovi <= 1:
        print("Ocjena: 3")
    elif bodovi >= 0.6 and bodovi <= 1:
        print("Ocjena: 2")
    elif bodovi >= 0.5 and bodovi <= 1:
        print("Ocjena: 1")
except:
    print("Neispravan unos")