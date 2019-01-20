import pandas as pd
import glob

Files = glob.glob('*.csv')
for f in Files:
    print(pd.read_csv(f, names=('ch','time','width')))