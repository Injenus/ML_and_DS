import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_N = 1
OUTPUT_N = 1
HIDDEN_N = 7
n_hidden_layers = [9, 15]


def getName(arg):
    gl = globals().copy()
    for i in gl:
        if gl[i] is arg:
            return (i)


def tensor_to_csv(tensor):
    ten_np = tensor.numpy()
    df = pd.DataFrame(ten_np)
    df.to_csv(getName(tensor), index=False)


def from_csv(file_name):
    data = torch.tensor((pd.read_csv(file_name)).values)
    data = data.type(torch.FloatTensor)
    return data


# x_train, x_test, y_train, y_test = 0, 0, 0, 0
#
# x_train = from_csv(getName(x_train))
# y_train = from_csv(getName(y_train))
# x_test = from_csv(getName(x_test))
# y_test = from_csv(getName(y_test))


x_train = torch.rand(1500) * 6 * math.pi
y_train = np.cos(x_train) + np.arctan(x_train)
x_test = torch.rand(1500) * 6 * math.pi
x_test.numpy().sort()
y_test = np.cos(x_test) + np.arctan(x_test)

x_train = x_train.unsqueeze(1)
y_train = y_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)
y_test = y_test.unsqueeze(1)


class Network(torch.nn.Module):
    def __init__(self, hidden_neurons):
        super(Network, self).__init__()
        # # 1 слой, принимающий сигнал
        # self.fc1 = torch.nn.Linear(INPUT_N, HIDDEN_N)
        # # 2  функция сигмоиды для промежуточного слоя
        # self.act1 = torch.nn.Sigmoid()
        # # self.act2 = torch.nn.Sigmoid()
        # # 3 фнукция актвиации для выходного слоя
        # self.fc2 = torch.nn.Linear(HIDDEN_N, OUTPUT_N)
        self.l0 = torch.nn.Linear(INPUT_N, n_hidden_layers[0])
        self.w1 = torch.nn.Sigmoid()
        self.l1 = torch.nn.Linear(n_hidden_layers[0], n_hidden_layers[1])
        self.w2 = torch.nn.Sigmoid()
        self.l2 = torch.nn.Linear(n_hidden_layers[1], OUTPUT_N)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.act1(x)
        # # x = self.act2(x)
        # x = self.fc2(x)
        x = self.l0(x)
        x = self.w1(x)
        x = self.l1(x)
        x = self.w2(x)
        x = self.l2(x)
        return x


def loss(pred, target):
    ans = (pred - target) ** 2
    return ans.mean()


def predict(net, x, y):
    y_pred = net.forward(x).detach()
    plt.clf()
    plt.title("Epoсh-%d" % i)
    plt.plot(x.numpy(), y.numpy(), c='b', label='Ground truth')
    plt.plot(x.numpy(), y_pred.numpy(), c='r', label='Rediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')


plt.title('Target')
plt.plot(x_train, y_train)
plt.savefig('images/Target.png')

if __name__ == '__main__':
    our_network = Network(HIDDEN_N)
    optimizer = torch.optim.Adam(our_network.parameters(), 0.01)

    for i in range(25001):
        optimizer.zero_grad()
        y_pred = our_network.forward(x_train)
        loss_val = loss(y_pred, y_train)
        loss_val.backward()
        optimizer.step()
        if (i % 5000 == 0):  # в начале каждой эпохи
            y_pred = our_network.forward(x_test)
            print(loss(y_pred, y_test) ** 0.5)
            predict(our_network, x_test, y_test)
            plt.savefig("images/Epoсh-%d.png" % i)
