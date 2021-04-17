import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


print(step_function(0.5))
print(step_function(1))
print(step_function(0))


def step_function(x):
    y = x > 0
    y = y.astype(int)
    return y


print(step_function(np.array([1, 2, 3, 0])))


def step_function(x):
    return np.array(x > 0, dtype=int)


x = np.arange(-5, 5, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.show()