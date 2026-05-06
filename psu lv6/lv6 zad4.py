import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

img = mpimg.imread('example_grayscale.png')

if len(img.shape) == 3:
    img = img.mean(axis=2)

X = img.reshape(-1, 1)

K = 10
kmeans = KMeans(n_clusters=K, n_init=10)
kmeans.fit(X)

vrijednosti = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_

slika_komrpesirana = vrijednosti[labels].reshape(img.shape)

plt.figure()
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.figure()
plt.imshow(slika_komrpesirana, cmap='gray')
plt.title(f"Kvantizirana ({K} klastera)")
plt.show()

# veci K -> bolja slika, manji K -> vidljivi efekt

# kompresijski omjer za K=10:
# original: 8 bita po pikselu (256 razina sive)
# komprimirano: 4 bita po pikselu (samo indeks klastera) rezultira losiju kvalitetu slike

n = img.shape[0] * img.shape[1]
original_bits = n * 8
compressed_bits = n * int(np.ceil(np.log2(K))) + K * 8  # pikseli + centri klastera


# podzadatak: KOLIKO SE KOMPRESIJE SLIKE MOŽE POSTIĆI AKO SE KORISTE 10 KLASTERA
# broj piksela
num_pixels = img.shape[0] * img.shape[1]

# original: 256 razina sive -> 8 bita po pikselu
original_bits = num_pixels * 8

# kompresija:
# svaki piksel sada treba log2(K) bita
K = 10
bits_per_pixel = np.ceil(np.log2(K)) # zaokruživanje na cijeli broj bitova

compressed_bits = num_pixels * bits_per_pixel

# spremljeni centri klastera
compressed_bits += K * 8 # svaki centar = 8 bita

# omjer kompresije
compression_ratio = original_bits / compressed_bits

print(f"Original veličina (bitovi): {original_bits}")
print(f"Kompresirana veličina (bitovi): {compressed_bits}")
print(f"Kompresijski omjer: {compression_ratio:.2f}x")