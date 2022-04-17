import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# Количество нейронов на втором скрытом слое(?)
secLNn = 3
# Количетсво тестовых данных
quan = 10000
# Создание случайных входных данных
Np = 40
x = 2 * math.pi * (0.5 - torch.rand(Np))
y = 2 * math.pi * (0.5 - torch.rand(Np))
plt.plot(x.numpy(), y.numpy(), 'o', c='k')
plt.savefig('mygraph.jpg')
print(type(x))
print(x)
# Создание выходных данных
k = 4
f = lambda x: k * torch.sin(x)
z = np.sign(y - f(x))

i1_by_nn = [i for i, x in enumerate(z) if x <= 0]
i2_by_nn = [i for i, x in enumerate(z) if x > 0]

plt.plot(x.numpy()[i1_by_nn], y.numpy()[i1_by_nn], 'o', c='g')
plt.plot(x.numpy()[i2_by_nn], y.numpy()[i2_by_nn], 'o', c='b')
plt.savefig('mygraph2.jpg')

# Создание обучающей выборки
x = x.unsqueeze(1)
y = y.unsqueeze(1)
I = torch.cat((x, y), dim=1)
targVals = z.unsqueeze(1)


# Создание НС
class Network(torch.nn.Module):
    def __init__(self, hidden_neurons):
        super(Network, self).__init__()
        # 1 слой, принимающий сигнал
        self.fc1 = torch.nn.Linear(2, hidden_neurons)
        # 2  функция сигмоиды для вторго слоя
        self.act1 = torch.nn.Sigmoid()
        #
        self.fc2 = torch.nn.Linear(hidden_neurons, secLNn)
        self.act2 = torch.nn.Sigmoid()
        # 3 фнукция актвиации для выходного слоя
        self.fc3 = torch.nn.Linear(secLNn, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


our_network = Network(10)

# Обучение НС
optimizer = torch.optim.Adam(our_network.parameters(), 0.01)


def loss(pred, target):
    ans = (pred - target) ** 2
    return ans.mean()


for i in range(500):
    optimizer.zero_grad()
    y_pred = our_network.forward(I)
    loss_val = loss(y_pred, targVals)
    loss_val.backward()
    optimizer.step()
    if (i % 100 == 0):
        y_pred = our_network.forward(I)
        print((((y_pred - targVals) ** 2).mean()) ** 0.5)

# Тестирвоание НС
xs = torch.linspace(-math.pi, math.pi, steps=int(quan ** 0.5))
ys = torch.linspace(-math.pi, math.pi, steps=int(quan ** 0.5))
x_test, y_test = torch.meshgrid(xs, ys, indexing='xy')

plt.clf()
plt.plot(x_test.numpy(), y_test.numpy(), '.', c='k')
plt.savefig('2-test-x.jpg')

x_test = torch.flatten(x_test).unsqueeze(1)
y_test = torch.flatten(y_test).unsqueeze(1)
I_test = torch.cat((x_test, y_test), dim=1)
# расчёт сети на тестовых данных
out_nn = our_network.forward(I_test)
# определение индексов точек разных классов и визуализация
# 1 класса
i1_by_nn = [i for i, x in enumerate(out_nn) if x <= 0]
# 2 класса
i2_by_nn = [i for i, x in enumerate(out_nn) if x > 0]
plt.clf()
plt.plot(x_test.detach().numpy()[i1_by_nn], y_test.detach().numpy()[i1_by_nn],
         '.', c='g')
plt.plot(x_test.detach().numpy()[i2_by_nn], y_test.detach().numpy()[i2_by_nn],
         '.', c='b')

kd = 100
x_real = [i / kd for i in range(int(-math.pi * kd), int(math.pi * kd)) ]
y_real = [k * math.sin(el) for el in x_real]
plt.plot(x_real, y_real, c='r', label='Target')
plt.legend()
plt.savefig('2-test-y.jpg')

# Численная оценка
err_cnt = 0
for i, point in enumerate(I_test):
    real_class = -1 if point[1] <= f(point[0]) else 1
    if real_class * out_nn[i] < 0:
        err_cnt += 1
print('Неверно распознанных входных данных {} ед. - {:.3f}%'.format(err_cnt,
                                                                    err_cnt / len(
                                                                        I_test) * 100))
