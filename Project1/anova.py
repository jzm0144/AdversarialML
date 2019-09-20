import scipy.stats as stats
import pandas as pd 


df = pd.read_csv("Results_Log.csv")

Elitist_GA             = df['Elitist__Generational_GA']
Steady_State_GA        = df['Steady_State_GA']
Steady_Generational_GA = df['Steady_Generational_GA']
mu_plus_mu_GA          = df['µ+µ_GA'] 

print("Means  = ", df.mean(axis=0),"\n\n")



(t, p) = stats.f_oneway(Elitist_GA, Steady_State_GA, Steady_Generational_GA, mu_plus_mu_GA)
print("t value = ", t, '   and p value = ',p)



(t, p) = stats.f_oneway(Steady_State_GA, Steady_Generational_GA, mu_plus_mu_GA)
print("t value = ", t, '   and p value = ',p)

(t, p) = stats.f_oneway(Steady_State_GA, Steady_Generational_GA)
print("t value = ", t, '   and p value = ',p)
