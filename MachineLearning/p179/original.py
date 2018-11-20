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


#---基底関数---
def model_A(x, w):
	return w[0] * x**3  + w[1] * x**2 + w[2] * x + w[3]

"""def mse_line(x, t, w):
    y = w[0] * x + w[1]
    mse = np.mean((y - t)**2)
    return mse"""

def mse_model_A(w, x, t):
	y = model_A(x, w)
	mse = np.mean((y - t)**2)
	return mse

from scipy.optimize import minimize

def fit_model_A(w_init, x, t):
	res1 = minimize(mse_model_A, w_init, args=(x, t), method="powell")
	return res1.x


def show_model_A(w):
	xb = np.linspace(X_min, X_max, 100)
	y = model_A(xb, w)
	plt.plot(xb, y, c=[.5, .5, .5], lw=4) #RGB=[0.5, 0.5, 0.5]

plt.figure(figsize=(4, 4))
W_init = [1, 1, 1, 1]
W = fit_model_A(W_init, X, T)
show_model_A(W)
plt.plot(X, T, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min, X_max)
plt.grid(True)
mse = mse_model_A(W, X, T)
print('W=' + str(np.round(W, 1)))
print('SD={0: .2f} cm'.format(np.sqrt(mse)))
plt.show()