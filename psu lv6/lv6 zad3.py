import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from funkcija_6_1 import generate_data

X = generate_data(500, 1)

Z = linkage(X, method='ward') 

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Podaci")
plt.ylabel("Udaljenost")
plt.show()
