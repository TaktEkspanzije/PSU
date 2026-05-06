import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from funkcija_6_1 import generate_data

X = generate_data(500, 1)

inertia = []
for k in range(1, 21):
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(range(1, 21), inertia, marker='o')
plt.xlabel("Broj klastera K")
plt.ylabel("Kriterijska funkcija J")
plt.title("Lakat metoda")
plt.show()
