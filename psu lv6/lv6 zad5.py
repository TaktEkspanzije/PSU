import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

slika = mpimg.imread('example.png')

# ista logika kao za grayscale, samo su pikseli sad RGB
X = slika.reshape(-1, 3)

kmeans = KMeans(n_clusters=10, n_init=10)
kmeans.fit(X)

slika_kompresirana = kmeans.cluster_centers_[kmeans.labels_].reshape(slika.shape)

if (slika_kompresirana.max() <= 1.0):
    slika_kompresirana = (slika_kompresirana * 255).astype(np.uint8)
else:
    slika_kompresirana = slika_kompresirana.astype(np.uint8)

plt.figure()
plt.imshow(slika)
plt.title("Original")

plt.figure()
plt.imshow(slika_kompresirana)
plt.title("Kompresirana (10 boja)")
plt.show()