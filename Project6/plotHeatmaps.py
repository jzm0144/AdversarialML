import numpy as np
import pandas as pd
import ipdb as ipdb
import matplotlib.pyplot as plt


'''
df = pd.read_csv('allHeatmaps/0.csv')
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
'''

df = pd.read_csv('avgHeatmaps/1000.csv')
df = df.values[:,1:]
print(df.shape)

plt.figure()
plt.subplot(311);
plt.plot(df[:,0]);
plt.title('Mean Occlusion Heatmaps')
plt.ylabel("Heat Scores")
plt.subplot(312);
plt.plot(df[:,10]);
plt.subplot(313);
plt.plot(df[:,21]);
plt.xlabel('Features')
plt.show()

ipdb.set_trace()


