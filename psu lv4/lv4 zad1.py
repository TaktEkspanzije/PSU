import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error

def non_func(x):
	y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
	return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0,1,len(y))
    return y_noisy

x = np.linspace(1,10,100)
y_true = non_func(x)
y_measured = add_noise(y_true)

plt.figure(1)
plt.plot(x,y_measured,'ok',label='mjereno')
plt.plot(x,y_true,label='stvarno')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 4)
plt.show()

np.random.seed(12)
indeksi = np.random.permutation(len(x))
indeksi_train = indeksi[0:int(np.floor(0.7*len(x)))]
indeksi_test = indeksi[int(np.floor(0.7*len(x)))+1:len(x)]

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

xtrain = x[indeksi_train]
ytrain = y_measured[indeksi_train]

xtest = x[indeksi_test]
ytest = y_measured[indeksi_test]

plt.figure(2)
plt.plot(xtrain,ytrain,'ob',label='train')
plt.plot(xtest,ytest,'or',label='test')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 4)
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(xtrain,ytrain)

print('Model je oblika y_hat = Theta0 + Theta1 * x')
print('y_hat = ', linearModel.intercept_, '+', linearModel.coef_, '*x')

ytest_p = linearModel.predict(xtest)
MSE_test = mean_squared_error(ytest, ytest_p)

plt.figure(3)
plt.plot(xtest,ytest_p,'og',label='predicted')
plt.plot(xtest,ytest,'or',label='test')
plt.legend(loc = 4)

x_pravac = np.array([1,10])
x_pravac = x_pravac[:, np.newaxis]
y_pravac = linearModel.predict(x_pravac)
plt.plot(x_pravac, y_pravac)
plt.show()


"""
non_func(x) 
je funkcija f(x) pod (4-3), a add_noise(y) je funkcija koja dodaje šum na stvarne vrijednosti funkcije odnosno epsilon ono e iz pdf-a.
Regresija.
linearModel = lm.LinearRegression() je funkcija koja kreira model linearne regresije, a linearModel.fit(xtrain,ytrain) je funkcija koja trenira model na podacima xtrain i ytrain.
linearModel.intercept_ i linearModel.coef_ su parametri modela, odnosno Theta0 i Theta1.
Ovaj linija plus prethodna je višedimenzionalna linearna regresija, (4-7) tj. Definiranje modela kao linearne funkcije.
linearModel.fit(xtrain,ytrain) je funkcija koja trenira model na podacima xtrain i ytrain.
ytest_p = linearModel.predict(xtest) računa procjenu y kapica za testne podatke.
MSE_test = mean_squared_error(ytest, ytest_p) implementira formulu za srednju kvadratnu pogrešku (4-14) i računa MSE za testne podatke. Vrednovanje modela.
x = x[:, np.newaxis] pretvara obični niz u stupčasti vektor/matricu (n x 1) kako bi bio kompatibilan s matričnim izračunima u scikit-learn.
"""