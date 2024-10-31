import numpy as np

true = np.load("true.npy")
pred = np.load("pred.npy")
conf = np.load("conf.npy")
error = np.abs(true-pred)

print(true.shape)
print(pred.shape)
print(conf.shape)
print(error.shape)

print('--- error and confidence')
conf = conf.reshape(-1,1)
error = error.reshape(-1,1)
error_confidence = np.concatenate((conf, error),axis=1)

corr = np.corrcoef(error_confidence, rowvar=False)
print(corr)
