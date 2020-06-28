#!/usr/bin/env python
# coding: utf-8

# In[12]:


import glob
import os
import numpy as np

dataset = 'D:\python\GCN\DeepGraphClustering\data\MUTAG'
gmls = glob.glob(dataset + '\*.gml')

def idx_format(i):
    i1 = int(i/100)
    i2 = int(i%100/10)
    i3 = int(i%10)
    return '{}{}{}'.format(i1, i2, i3)

with open(dataset + '\Labels.txt') as f:
    ls = f.readlines()
    labels = np.array([int(l.rstrip()) for l in ls])
labels_ = np.copy(labels)
n_of_y = 0
for l in labels:
    if(l == 1):
        n_of_y += 1

map = {}
next_i_y, next_i_n = 0, 0
for i in range(len(labels)):
    if(labels[i] == 1):
        map[i] = next_i_y
        next_i_y += 1
    else:
        map[i] = n_of_y + next_i_n
        next_i_n += 1
        
for before, after in enumerate(map.values()):
    labels_[after] = labels[before]

'''
for i in range(188):
    os.rename('{}\{}_buff.gml'.format(dataset, i), '{}\{}.gml'.format(dataset, i))
'''
with open(dataset + '\Labels_.txt', 'w') as w:
    for l in labels_:
        w.write('{}\n'.format(l))


# In[ ]:




