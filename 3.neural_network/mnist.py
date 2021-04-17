import tensorflow
from keras.datasets import mnist
import numpy as np
from PIL import Image
import pickle


def sigmoid(x):
    x = x - np.max(x)
    return np.exp(x) / (1 + np.exp(-abs(x)))


def softmax(x):
    x = x - np.max(x)
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape([x_train.shape[0], 28 * 28])  # １次元に変換
    x_test = x_test.reshape([x_test.shape[0], 28 * 28])  # １次元に変換
    return x_test, y_test


def image_show(img):
    pil_image = Image.fromarray(img)
    pil_image.show()


def init_network():
    with open('./sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x_test, y_test = get_data()
network = init_network()
accuracy_count = 0
for i in range(len(x_test)):
    y = predict(network, x_test[i])
    p = np.argmax(y)
    if p == y_test[i]:
        accuracy_count += 1

print(float(accuracy_count) / len(x_test))

print(len(x_test))

batch_size = 100
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_count += np.sum(p == y_test[i:i + batch_size])

print(float(accuracy_count) / len(x_test))

print(len(x_test))
