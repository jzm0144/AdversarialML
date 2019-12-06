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

lsvm = np.round(lsvm_mean[:-1])
rbfsvm = np.round(rbfsvm_mean[:-1])
mlp = np.round(mlp_mean[:-1])
mask = np.zeros(lsvm.shape)

for i in range(len(lsvm)):
    if (lsvm[i] == 1 or rbfsvm[i] == 1) and mlp[i] == 1:
        mask[i] = 1


print('Mask = ', mask)
print("Mask Sum = ", sum(mask))

ipdb.set_trace()
Mask = pd.DataFrame({"mask":mask})

Mask.to_csv('Mask.csv')

