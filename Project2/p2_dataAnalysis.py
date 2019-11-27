import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Project2_Results_Log.csv")

Sync   = df['Sync'].values
aSync  = df['aSync'].values

print("Means  = ", np.mean(Sync), "   ", np.mean(aSync), "\n\n")


(t, p) = stats.ttest_ind(Sync, aSync)
print("t value = ", t, '   and p value = ',p)


# Generating the BoxPlot-for-Results
data = [Sync, aSync]
plt.boxplot(data, labels= ["Sync", "aSync"], showmeans=True)
plt.title('Fitness values Particle Swarms Optimizers')
plt.ylabel("Best Fitnesses")
plt.show()