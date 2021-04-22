import numpy as np
import matplotlib.pylab as plt


def func(x):
    return x ** 2


def numerical_diff(x):
    h = 10e-50  # 数値の丸め込み誤差により値が0に
    return (func(x + h) - func(x)) / h


print(numerical_diff(3))


def numerical_diff(x):
    h = 1e-4
    return (func(x + h) - func(x)) / h  # 実際の接線の傾きとは誤差が生じる


print(numerical_diff(3))


def numerical_diff(x):
    h = 1e-4
    return (func(x + h) - func(x - h)) / (2 * h)  # 中心差分を取ることで、より誤差を減らす


print(numerical_diff(3))


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()


def function_2(x):
    return x[0] ** 2 + x[1] ** 2

# p.105