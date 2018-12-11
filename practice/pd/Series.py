import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

s1 = pd.Series([3, 3, 4], index=['b', 'c', 'a'])
s2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

print(s1)
print(s2)
print(s1 * s2)
print(s1>3)