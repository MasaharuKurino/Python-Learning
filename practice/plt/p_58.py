import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 2) * x * (x + 2)

x = np.linspace(-3, 3, 10)
y = f(x)

plt.plot(x, y)
plt.show()