import numpy as np
# a= np.array([0,-1,-2,3,-4,-5,6,7,8])
# b = np.where(a < 0, a*a**2, a)
# c = np.where(a < 0)
# print(a)
# print(b)
# print(c[0])

# t1 = np.array([3,2,1])
# t2 = np.array([1,2,3])
# t3 = t1*t2
# print(t3)
a = np.array([1,2,4,5,6])
b = np.array([2,3,5,7,8])
c = np.union1d(a,b)
print(c)
a= np.array([0,1,2,3,4,5,6,7,8])
f = a[[1,3,5]]
b = np.where(a < 0, a*a**2, a)
c = np.where(a < 0)
d = np.array([])
e = np.union1d(d,c)
print(f)
print(a**2)
