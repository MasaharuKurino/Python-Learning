import numpy as np
import pandas as pd
import glob

M = np.zeros((12,12))

Files = glob.glob('*.csv')
for f in Files:
    df = pd.read_csv(f, names=('ch','time','width')) #データフレーム
    print(df['ch'].value_counts())