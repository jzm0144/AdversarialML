import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Results_Log.csv")

Elitist_GA             = df['EGA'].values
Steady_State_GA        = df['SSGA'].values
Steady_Generational_GA = df['SGGA'].values
mu_plus_mu_GA          = df['µ_plus_µ_GA'] .values

print("Means  = ", np.mean(Elitist_GA), "   ", np.mean(Steady_State_GA), "   ", np.mean(Steady_Generational_GA), "   ", np.mean(mu_plus_mu_GA),"\n\n")



(t, p) = stats.f_oneway(Elitist_GA, Steady_State_GA, Steady_Generational_GA, mu_plus_mu_GA)
print("t value = ", t, '   and p value = ',p)



(t, p) = stats.f_oneway(Steady_State_GA, Steady_Generational_GA, mu_plus_mu_GA)
print("t value = ", t, '   and p value = ',p)



(t, p) = stats.ttest_ind(Steady_State_GA, Steady_Generational_GA)
print("t value = ", t, '   and p value = ',p)


# Generating the BoxPlot-for-Results
data = [Elitist_GA, Steady_State_GA, Steady_Generational_GA, mu_plus_mu_GA]
plt.boxplot(data, labels= ['Elitist','SSGA','SGGA','µ+µ'], showmeans=True)
plt.title('Fitness values of the Genetic Algorithms')
plt.ylabel("Best Fitnesses")
plt.show()

