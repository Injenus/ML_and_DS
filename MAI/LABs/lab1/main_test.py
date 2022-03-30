import math
import torch
import numpy as np
import matplotlib.pyplot as plt

HID_N = 3

# dx = 20  # кол-во точек на PI
# x_tst = [i / 1000 for i in
#          range(0, int(6 * math.pi * 1000), int(math.pi * 1000 / dx))]
# x_trn = [i / 1000 for i in
#          range(0, int(6 * math.pi * 1000), int(math.pi * 1000 / dx))]
# print(len(x_tst), len(x_trn))
# x_test = torch.tensor(x_tst)
# x_train = torch.tensor(x_trn)

x_train = torch.rand(1000) * 6 * math.pi
x_test = torch.rand(333) * 6 * math.pi

x_test.numpy().sort()
y_train = np.cos(x_train) + np.arctan(x_train)
y_test = np.cos(x_test) + np.arctan(x_test)

class Network(torch.nn.Module):
    def __init__(self, hidden_neurons):
        super(Network, self).__init__()
        # 1 слой, принимающий сигнал
        self.fc1 = torch.nn.Linear(1, hidden_neurons)
        # 2  функция сигмоиды для промежуточных слоёв слоя
        self.act1 = torch.nn.Sigmoid()
        # self.act2 = torch.nn.Sigmoid()
        # 3 фнукция актвиации для выходного слоя
        self.fc2 = torch.nn.Linear(hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        # x = self.act2(x)
        x = self.fc2(x)
        return x


def loss(pred, target):
    ans = (pred - target) ** 2
    return ans.mean()


def predict(net, x, y):
    y_pred = net.forward(x).detach()
    plt.clf()
    plt.title("Epoсh №%d" % i)
    plt.plot(x.numpy(), y.numpy(), 'o', c='b', label='Ground truth')
    plt.plot(x.numpy(), y_pred.numpy(), 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')


plt.title('Target')
plt.plot(x_test, y_test, 'o')
plt.savefig('images/Target.png')

if __name__ == '__main__':
    our_network = Network(HID_N)
    optimizer = torch.optim.Adam(our_network.parameters(), 0.004)

    x_train = x_train.unsqueeze(1)
    y_train = y_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    for i in range(40001):
        optimizer.zero_grad()
        y_pred = our_network.forward(x_train)
        loss_val = loss(y_pred, y_train)
        loss_val.backward()
        optimizer.step()
        if (i % 500 == 0):  # в начале каждой эпохи
            y_pred = our_network.forward(x_test)
            print(loss(y_pred, y_test) ** 0.5, i)
            predict(our_network, x_test, y_test)
            plt.savefig("images/Epoсh №%d.png" % i)
    print('final', loss(y_pred, y_test) ** 0.5)
