import numpy as np
import matplotlib.pyplot as plt

#元データ生成
np.random.seed(seed=0)
X_n = 30
X = np.zeros(X_n)
T = np.zeros(X_n, dtype=np.uint8) #unit8 = 8ビット符号なし整数
Dist_s = [0.4, 0.8]
Dist_w = [0.8, 1.6]
Pi = 0.5
for n in range(X_n):
    wk = np.random.rand()
    T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi)
    X[n] = np.random.rand() * Dist_w[T[n]] + Dist_s[T[n]]


def logistic(x, w):
    return 1 / (1 + np.exp(-(w[0] * x + w[1])))

#平均交差エントロピー誤差
def cee_logistc(w, x, t):
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + (1 - t[n]) * np.log(1 - y[n]))
    cee /= X_n
    return cee

#平均交差エントロピー誤差の微分
def dcee_logistic(w, x, t):
    y = logistic(x, w)
    dcee = np.zeros(2)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n]) * X[n]
        dcee[1] = dcee[1] + (y[n] - t[n])
    dcee /= X_n
    return dcee


from scipy.optimize import minimize
def fit_logistic(w_init, x, t):
    res1 = minimize(cee_logistc, w_init, args=(x, t),
                    jac=dcee_logistic, method="CG")
    return res1.x


#--ここから表示処理------------------------------

X_min = 0
X_max = 2.5

#Wに応じてロジスティック曲線表示
def show_logistic(w):
    xb = np.linspace(X_min, X_max, 100)
    y = logistic(xb, w)
    plt.plot(xb, y, color='gray', linewidth=4)
    
    i = np.min(np.where(y > 0.5))
    B = (xb[i - 1] + xb[i]) / 2 #決定境界
    plt.plot([B, B], [-0.5, 1.5], color='K', linestyle='--')
    plt.grid(True)

#元データ表示
def show_data1(x, t):
    X_col = ['cornflowerblue', 'gray']
    K = np.max(t) + 1
    for k in range(K):
        plt.plot(x[t == k], t[t == k], X_col[k], alpha=0.5, linestyle='none', marker='o')
    plt.grid(True)
    plt.ylim(-0.5, 1.5)
    plt.xlim(X_min, X_max)
    plt.yticks([0, 1])

plt.figure(1, figsize=(4, 4))
W_init = [1, -1] #Wの初期値
W = fit_logistic(W_init, X, T) #W最小化
print("w0 = {0:.2f}, w1 = {1:.2f}".format(W[0], W[1])) 
show_logistic(W) #最小のWを与えてロジスティック曲線表示
show_data1(X, T)
plt.show()