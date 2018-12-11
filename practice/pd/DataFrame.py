import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

d1 = pd.DataFrame([[1, 2], [1, 1], [3, 5]], index=['a', 'b', 'c'], columns=['x', 'y'])
print(d1)

s1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s2 = pd.Series([10, 40, 120], index=['b', 'c', 'd'])
d2 = pd.DataFrame({'col1': s1, 'col2': s2})
print(d2)
print(d2.describe())
print(d2.describe()['col1']['mean'])
print(d2.describe().col1)
print(d2.col1)