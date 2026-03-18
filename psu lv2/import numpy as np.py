import numpy as np
import matplotlib.pyplot as plt


x=np.array([1,3,3,2,1])
y=np.array([1,1,2,2,1])

#x = np.linspace(0, 4, num=30)

plt.plot(x, y, 'b', linewidth=1, marker=".", markersize=5)

plt.axis([0,4,0,4])

plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')

plt.xlim(0, 4)
plt.ylim(0, 4)

plt.show()







































