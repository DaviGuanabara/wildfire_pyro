import numpy as np
import matplotlib.pyplot as plt

lr1 = (1/(1+0.1*(700-np.arange(0,400)))) * 0.5

lr_first_half = lr1[-1]

lr2 = (0.99  ** (np.arange(400,1000)-400)) * lr_first_half

lr = np.concatenate((np.reshape(lr1, (-1,1)), np.reshape(lr2, (-1,1))), 0)

plt.figure()
plt.plot(np.arange(0, lr.shape[0]), lr)
plt.show()

