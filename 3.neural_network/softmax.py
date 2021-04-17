import numpy as np


def softmax(x):
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)


a1 = np.array([1010, 1000, 990])
y1 = softmax(a1)  # オーバーフローにより正しく計算されない
print(y1)

c = np.max(a1)
print(c)
y1 = softmax(a1 - c)  # ソフトマックスは足し算引き算により結果が変わらないため、要素の最大値で引くことでオーバーフローを防ぐ
print(y1)
