import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipdb as ipdb



lsvm = pd.read_csv('lsvm.csv')
rbfsvm = pd.read_csv('rbfsvm.csv')
mlp = pd.read_csv('mlp.csv')


lsvm_mean = lsvm.mean()
rbfsvm_mean = rbfsvm.mean()
mlp_mean = mlp.mean()

lsvm_mean = lsvm_mean.values
rbfsvm_mean = rbfsvm_mean.values
mlp_mean = mlp_mean.values


lsvm_std = lsvm.std()
rbfsvm_std = rbfsvm.std()
mlp_std = mlp.std()

lsvm_std = lsvm_std.values
rbfsvm_std = rbfsvm_std.values
mlp_std = mlp_std.values

plt.figure(1)
plt.subplot(311)
plt.plot(lsvm_mean[:-1])
plt.plot(lsvm_std[:-1], marker = '.')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
plt.ylabel('LSVM')

plt.subplot(312)
plt.plot(rbfsvm_mean[:-1])
plt.plot(rbfsvm_std[:-1], marker = '.')
plt.ylabel('RBFSVM')

plt.subplot(313)
plt.plot(mlp_mean[:-1])
plt.plot(mlp_std[:-1], marker = '.')
plt.xlabel('Features')
plt.ylabel('MLP')

plt.show()


print("Mean Fitness Values:")
print("LSVM: ", lsvm_mean[-1], "    RBFSVM: ",rbfsvm_mean[-1], "   MLP: ", mlp_mean[-1])
print("\nStandard Deviation of Fitness Values:")
print("LSVM: ", lsvm_std[-1], "    RBFSVM: ",rbfsvm_std[-1], "   MLP: ", mlp_std[1])




# Mean Vector Lengths
lsvm = lsvm.values[:, :-1]
rbfsvm = rbfsvm.values[:, :-1]
mlp = mlp.values[:, :-1]

temp = lsvm.sum(axis=1)
lsvm_vec_Length = temp.mean()

temp = rbfsvm.sum(axis=1)
rbfsvm_vec_Length = temp.mean()

temp = mlp.sum(axis=1)
mlp_vec_Length = temp.mean()

print("\n\nMean Feature Vector Lengths")
print("LSVM   = ", lsvm_vec_Length)
print("RBFSVM = ", rbfsvm_vec_Length)
print("MLP    = ", mlp_vec_Length)
