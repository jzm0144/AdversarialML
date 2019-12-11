import numpy as np
import pandas as pd
import ipdb as ipdb
import matplotlib.pyplot as plt



df = pd.read_csv('allHeatmaps/0.csv')
df = df.values[:,1:]


plt.figure(1)
plt.plot(df[:,0])
plt.ylabel("Heat Scores")
plt.ylim(0, 1)
plt.xlabel('features')
plt.title('Occlusion Map (1001)')
plt.show()



df1 = pd.read_csv('avgHeatmaps/1000.csv')
df1 = df1.values[:,1:]
df2 = pd.read_csv('avgHeatmaps/1005.csv')
df2 = df2.values[:,1:]
df3 = pd.read_csv('avgHeatmaps/1008.csv')
df3 = df3.values[:,1:]


plt.figure(2)
plt.subplot(311);
plt.plot(df1[:,0]);
plt.ylim(0, 1)
plt.title('Mean Occlusion Heatmaps')
plt.subplot(312);
plt.plot(df2[:,5]);
plt.ylim(0, 1)
plt.ylabel("Heat Scores")
plt.subplot(313);
plt.plot(df3[:,8]);
plt.ylim(0, 1)
plt.xlabel('Features')
plt.show()

ipdb.set_trace()


