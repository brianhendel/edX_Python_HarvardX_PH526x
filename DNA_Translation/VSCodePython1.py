import numpy as np
import matplotlib.pyplot as plt
import random

x = np.random.normal(0,1, (2,100))
X = np.cumsum(x, axis=1)

x_0 = np.array([[0],[0]])
X = np.concatenate((x_0, X), axis=1)



plt.plot(X[0], X[1], "ro-")
plt.show()

'''
Using Numpy Random Module - 2.4.4

X = np.random.randint(1,7,(10000,10))
Y = np.sum(X, axis=1)
plt.hist(Y)
np.sum(np.random.randint(1,7,(100,10)), axis=0)
'''

'''
#Examples with Randomness - 2.4.2
rolls = []
for k in range(10000):
    rolls.append(random.choice([1,2,3,4,5,6]))
plt.hist(rolls, bins=np.linspace(0.5, 6.5, 7))
ys = []

for rep in range(10000):
    y = 0
    for k in range(10):
        x = random.choice([1,2,3,4,5,6])
        y += x
    ys.append(y)
plt.hist(ys)

'''


'''
Histograms in Python - 2.3.4
x = np.random.normal(size=1000)
plt.hist(x, normed=True, bins=np.linspace(-5,5,21))

x = np.random.gamma(2,3,100000)
plt.figure()
plt.subplot(221)
plt.hist(x, bins=30)
plt.subplot(222)
plt.hist(x, bins=30, normed=True)
plt.subplot(223)
plt.hist(x, bins=30, cumulative=30)
plt.subplot(224)
plt.hist(x, bins=30, normed=True, cumulative=True, histtype='step')

'''

'''
Plotting in Python - 2.3.3
x = np.linspace(0, 10, 20)
y1 = x**2
y2 = x**1.5

plt.plot(x, y1, "bo-", linewidth=2, markersize=8, label="First")
plt.plot(x, y2, "gs-", linewidth=2, markersize=8, label="Second")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.axis([-0.5, 10.5, -5, 105])

plt.legend(loc="upper left")

plt.savefig("myplot.pdf")
'''