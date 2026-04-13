import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")

img = img[:,:,0].copy()

img_bright = np.clip(img + 0.3, 0, 1)

img_rot = np.rot90(img, k=-1)

img_flip = np.fliplr(img)

img_res = img[::10, ::10]

img_part = np.zeros_like(img)
sirina = img.shape[1]
pocetak = sirina // 4
kraj = 2 * (sirina // 4)
img_part[:, pocetak:kraj] = img[:, pocetak:kraj]

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(2, 3, 2)
plt.imshow(img_bright, cmap="gray")
plt.title("Posvijetljeno")

plt.subplot(2, 3, 3)
plt.imshow(img_rot, cmap="gray")
plt.title("Rotirano 90°")

plt.subplot(2, 3, 4)
plt.imshow(img_flip, cmap="gray")
plt.title("Zrcaljeno")

plt.subplot(2, 3, 5)
plt.imshow(img_res, cmap="gray")
plt.title("Manja rezolucija")

plt.subplot(2, 3, 6)
plt.imshow(img_part, cmap="gray")
plt.title("2. četvrtina")

plt.tight_layout()
plt.show()