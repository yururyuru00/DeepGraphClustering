import numpy as np

label = np.loadtxt('../cora/DANElabel.csv')
label = np.array([np.argmax(l) for l in label])
np.savetxt('../cora/DANElabel_vec.csv', label, fmt='%d')