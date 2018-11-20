#---p146---

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

"""np.random.seed(seed=1)
X_min = 4
X_max = 30
X_n = 16
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170, 108, 0.2]
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X)\
    + 4 * np.random.randn(X_n)
np.savez('ch5_data.npz', X=X, X_min=X_min, X_max=X_max, X_n=X_n, T=T)"""

outfile = np.load('ch5_data.npz')
X = outfile['X']
X_min = outfile['X_min']
X_max = outfile['X_max']
X_n = outfile['X_n']
T = outfile['T']


#---ガウス関数---
def gauss(x, mu, s):
	return np.exp(-(x - mu)**2 / (2 * s**2))

def gauss_func(w, x):
	m = len(w) - 1
	mu = np.linspace(5, 30, m)
	s = mu[1] - mu[0]
	y = np.zeros_like(x)
	for j in range(m):
		y = y + w[j] * gauss(x, mu[j], s)
	y = y + w[m]
	return y

"""def mse_line(x, t, w):
    y = w[0] * x + w[1]
    mse = np.mean((y - t)**2)
    return mse"""

def mse_gauss_func(x, t, w):
	y = gauss_func(w, x)
	mse = np.mean((y - t)**2)
	return mse

def fit_gauss_func(x, t, m):
	mu = np.linspace(5, 30, m)
	s = mu[1] - mu[0]
	n = x.shape[0]
	psi = np.ones((n, m+1))
	for j in range(m):
		psi[:, j] = gauss(x, mu[j], s)
	psi_T = np.transpose(psi)

	b = np.linalg.inv(psi_T.dot(psi))
	c = b.dot(psi_T)
	w = c.dot(t)
	return w

def show_gauss_func(w):
	xb = np.linspace(X_min, X_max, 100)
	y = gauss_func(w, xb)
	plt.plot(xb, y, c=[.5, .5, .5], lw=4)

M = 4
W = fit_gauss_func(X, T, M)
show_gauss_func(W)
plt.plot(X, T, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.grid(True)
mse = mse_gauss_func(X, T, W)
print('W=' + str(np.round(W, 1)))
print('SD={0: .2f} cm'.format(np.sqrt(mse)))
plt.show()