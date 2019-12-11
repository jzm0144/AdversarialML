import numpy as np
import pandas as pd
import ipdb as ipdb
import matplotlib.pyplot as plt



df = pd.read_csv('results/1.csv')
df = df.values
print(df.shape)

plt.figure()
plt.subplot(211)
plt.plot(df[:,3])
plt.ylabel("Heat Scores")
plt.subplot(212)
plt.plot(df[:,4:-1])
plt.xlabel('features')
plt.show()

ipdb.set_trace()


