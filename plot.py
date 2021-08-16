import _pickle as pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline as spline
import matplotlib.cm as cm
import os
import json

show=7000

def smooth_cost(cost, smooth=20):
    length = len(cost)
    r_index = np.arange(0, length, smooth)
    index = np.arange(length)
    a = []
    for i in range(0, length, smooth):
        a.append(sum(cost[i:i + smooth]) / smooth)
    a = np.array(a)
    Mean_ = spline(r_index, a)(index)
    return Mean_[:show]


with open('./checkpoint/{}reward_{}_{}.json'.format('', 2, 5),'r') as load_f:
    load_dict = json.load(load_f)
data = np.array(load_dict)
print(data.shape)

plt.figure(figsize=(10,10))

plt.subplot(411)
plt.title('reward')
plt.plot(smooth_cost(data[:,0]))

plt.subplot(412)
plt.title('step')
plt.plot(smooth_cost(data[:,1]))

plt.subplot(413)
plt.title('reward/step')
plt.plot(smooth_cost(data[:, 0]/data[:,1]))

plt.subplot(414)
plt.title('win rate')
plt.plot(smooth_cost(data[:,2]))

plt.subplots_adjust(wspace=0, hspace=0.5)  
plt.savefig('result.png')
plt.show()
