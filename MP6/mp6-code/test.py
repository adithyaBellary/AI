import numpy as np

a = np.arange(8).reshape(4,2)
y = [0,0,1,1]
ones_ = np.ones(a.shape[0]).reshape(-1,1)
# a = np.concatenate((a,ones_),axis=1)
print(a)
# a = a / np.sum(a,axis=1,keepdims=True)
# print(np.sum(a,axis=1,keepdims=True))
a[range(4),y] -= 1
print(a)
print(np.maximum(0,1))