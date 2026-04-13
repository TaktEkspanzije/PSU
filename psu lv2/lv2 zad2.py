import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)


mpg = data[:, 0]
cyl = data[:, 1]
hp  = data[:, 3]
wt  = data[:, 5]

plt.figure(figsize=(10, 6))

plt.scatter(hp, mpg, s=wt*60, c='blue', alpha=0.6, edgecolors='black', label='Velicina tocke = tezina (wt)')

plt.title('Ovisnost potrosnje o konjskoj snazi')
plt.xlabel('Konjska snaga (hp)')
plt.ylabel('Potrošnja (mpg)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

print("Statistika za sve automobile:")
print(f"Minimalni mpg: {np.min(mpg):.2f}")
print(f"Maksimalni mpg: {np.max(mpg):.2f}")
print(f"Srednji mpg: {np.mean(mpg):.2f}")

mpg_6_cyl = mpg[cyl == 6]

print("\nStatistika za automobile sa 6 cilindara:")

if len(mpg_6_cyl) > 0:
    print(f"Minimalni mpg (6 cyl): {np.min(mpg_6_cyl):.2f}")
    print(f"Maksimalni mpg (6 cyl): {np.max(mpg_6_cyl):.2f}")
    print(f"Srednji mpg (6 cyl): {np.mean(mpg_6_cyl):.2f}")
else:
    print("Nema automobila s 6 cilindara.")