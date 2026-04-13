import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 3, 1])
y = np.array([1, 2, 2, 1, 1])

plt.figure(figsize=(6, 5)) # duljina pa visina

#plt.plot(x, y, 'bo-', linewidth=2) #'boja,oblik tocaka,tip linije'
plt.plot(x,y, c = 'blue',linewidth = 2,marker = 'o')
plt.title('Primjer')
plt.xlabel('x os')
plt.ylabel('y os')
plt.axis([0,4,0,4])
plt.show()