import numpy as np


A = np.arange(1, 5, 1)
print(A)

print(np.ndim(A))  # 次元数

print(A.shape)  # 要素数

print(A.shape[0])  # 要素数


B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)

print(np.ndim(B))  # 次元数

print(B.shape)  # [要素数, 次の次元の要素数]

print(B.shape[0])  # 要素数


C = np.array([[1, 2], [3, 4]])
D = np.array([[5, 6], [7, 8]])

print(np.dot(C, D))
print(np.dot(D, C))

E = np.array([[1, 2, 3], [4, 5, 6]])
F = np.array([[1, 2], [3, 4], [5, 6]])

print(np.dot(E, F))
# print(np.dot(E, C))   行列Eの１次元の要素数と行列Cの0次元の要素数が違うため計算ができない

X = np.array([1, 2])  # 2入力・3出力の際の入力
W = np.array([[1, 2, 3], [4, 5, 6]])  # 2入力・3出力の際の重み
Y = np.dot(X, W)  # 2入力・3出力の際の出力
print('出力Y')
print(Y)
