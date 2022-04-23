import ast
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import random

HidNeur = 9
testNum = 100
incIdeal = 0
incNoise = 19
# Загрузка текстового файла с образами
nameOfFile = 'prprob0.txt'

noisePoint = 18


def readData(fileName):
    text = open(fileName, 'r')
    data = np.array([ast.literal_eval(line.strip(',\n'))
                     for line in text if line.strip()])
    data = data.astype(np.float32)
    data = data.reshape(26, 35)
    return data


Nvar = 7  # вариант
Nsymbols = 10  # число образов
inp = torch.from_numpy(
    readData(nameOfFile)[Nvar:Nvar + Nsymbols, :])  # входы для Nsymbols букв
out_ideal = torch.from_numpy(
    np.eye(Nsymbols))  # эталонные входы для идеальных данных

# Создание НС - 35 входов, Nsymbols выходов
class Network(torch.nn.Module):
    def __init__(self, hidden_neurons):
        super(Network, self).__init__()
        # 1 слой, принимающий сигнал
        self.fc1 = torch.nn.Linear(35, hidden_neurons)
        # 2  функция сигмоиды для вторго слоя
        self.act1 = torch.nn.Sigmoid()
        # 3 фнукция актвиации для выходного слоя
        self.fc3 = torch.nn.Linear(hidden_neurons, Nsymbols)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc3(x)
        return x


our_network = Network(HidNeur)
networkByNoise = Network(HidNeur)


def dataAugmen(ds, size=1, isNoised=0):
    if size:
        if isNoised:
            ds_2 = torch.cat((ds, addNoise(
                torch.from_numpy(readData(nameOfFile)[Nvar:Nvar + Nsymbols, :]),
                noisePoint)), 0)
            for i in range(size - 1):
                ds_2 = torch.cat((ds_2, addNoise(
                    torch.from_numpy(readData(nameOfFile)[Nvar:Nvar + Nsymbols, :]),
                    noisePoint)), 0)
            return ds_2
        else:
            ds_2 = torch.cat((ds, ds), 0)
            for i in range(size - 1):
                ds_2 = torch.cat((ds_2, ds), 0)
            return ds_2
    else:
        return ds


# Обучение НС
def learning(network, data_train, data_val, size, isNoise, nameOfNetwork):
    optimizer = torch.optim.Adam(network.parameters(), 0.01)

    if isNoise:
        data_train = dataAugmen(data_train, size, isNoise)
    else:
        data_train = dataAugmen(data_train, size, 0)
    data_val = dataAugmen(data_val, size, 0)

    print(data_train.shape, data_val.shape)

    def loss(pred, target):
        # print(pred.shape, target.shape)
        ans = (pred - target) ** 2
        return ans.mean()

    print('LOSS of "{}"'.format(nameOfNetwork))
    for i in range(20000):
        optimizer.zero_grad()
        y_pred = network.forward(data_train)
        loss_val = loss(y_pred, data_val)
        loss_val.backward()
        optimizer.step()
        if i % 200 == 0:
            y_pred = network.forward(data_train)
            print((((y_pred - data_val) ** 2).mean()) ** 0.5)
            # predict(our_network, x_test, y_test)
            # plt.savefig("mygraph-%d.jpg"%i)


# Тестирование
def testing(network, data_test, mark=''):
    out_nn = network.forward(data_test)
    out_nn = out_nn.detach().numpy()
    if mark:
        plt.clf()
        plt.imshow(out_nn, cmap='RdPu', clim=(-1, 1))
        plt.savefig('conf_matrix_' + mark + ".jpg")
    return out_nn


# print(inp.shape,out_ideal.shape)
learning(our_network, inp, out_ideal, incIdeal, 0, 'Идеальная_по_идеальным')

# Обучение на зашумленных данных

# Отображение эталонных данных
def convTo3D(input, height, width):
    output = torch.empty(len(input), height, width)
    for k, el in enumerate(input):
        for i in range(height):
            for j in range(width):
                output[k][i][j] = el[j + i * width]
    return output


row = math.floor(Nsymbols ** 0.5)
column = math.ceil(Nsymbols ** 0.5)
for i in range(Nsymbols):
    plt.subplot(row, column, i + 1)
    plt.imshow(convTo3D(inp, 7, 5)[i], cmap='RdPu', clim=(-1, 1))
plt.savefig("ideal_symbols.jpg")


# Добавление шума в данные
def addNoise(tnsr, noiseNumPoint):
    indxs = [i for i in range(35)]
    for el in tnsr:
        indxPoint = random.sample(indxs, noiseNumPoint)
        for indx in indxPoint:
            # if el[indx]:
            #     el[indx] = 0.
            # else:
            #     el[indx] = 1.
            el[indx] = 1.
    return tnsr


inp_noise_train = addNoise(
    torch.from_numpy(readData(nameOfFile)[Nvar:Nvar + Nsymbols, :]), noisePoint)

inp_noise_test = addNoise(
    torch.from_numpy(readData(nameOfFile)[Nvar:Nvar + Nsymbols, :]), noisePoint)

# Обучение
learning(networkByNoise, inp_noise_train, out_ideal, incNoise, 1,
         'Шумовая_по_зашумленным')

# ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ
plt.clf()
plt.subplot(2, 2, 1)
plt.imshow(testing(our_network, inp), cmap='RdPu', clim=(-1, 1))
plt.title('Идеал. НН по идеал. данным')
plt.subplot(2, 2, 3)
plt.imshow(testing(our_network, inp_noise_test), cmap='RdPu', clim=(-1, 1))
plt.title('Идеал. НН по шум. данным')
plt.subplot(2, 2, 2)
plt.imshow(testing(networkByNoise, inp), cmap='RdPu', clim=(-1, 1))
plt.title('Шум. НН по идеал. данным')
plt.subplot(2, 2, 4)
plt.imshow(testing(networkByNoise, inp_noise_test), cmap='RdPu', clim=(-1, 1))
plt.title('Шум. НН по шум. данным')
plt.subplots_adjust(wspace=0.25, hspace=0.3)
plt.savefig('TOTAL_conf_matrxs.jpg')

for i in range(Nsymbols):
    plt.subplot(row, column, i + 1)
    plt.imshow(convTo3D(inp_noise_train, 7, 5)[i], cmap='RdPu', clim=(-1, 1))
plt.savefig('train_noised_symbols.jpg')

for i in range(Nsymbols):
    plt.subplot(row, column, i + 1)
    plt.imshow(convTo3D(inp_noise_test, 7, 5)[i], cmap='RdPu', clim=(-1, 1))
plt.savefig('test_noised_symbols.jpg')

# ГИСТОРГРАММЫ

confMatr_ideal_for_ideal = np.zeros((Nsymbols, Nsymbols))
confMatr_noise_for_ideal = np.zeros((Nsymbols, Nsymbols))

confMatr_ideal_for_noise = np.zeros((Nsymbols, Nsymbols))
confMatr_noise_for_noise = np.zeros((Nsymbols, Nsymbols))

for i in range(testNum):
    inp_noise_test = addNoise(
        torch.from_numpy(readData(nameOfFile)[Nvar:Nvar + Nsymbols, :]), noisePoint)

    confMatr_ideal_for_ideal = confMatr_ideal_for_ideal + testing(our_network,
                                                                  inp)

    confMatr_noise_for_ideal = confMatr_noise_for_ideal + testing(our_network,
                                                                  inp_noise_test)

    confMatr_ideal_for_noise = confMatr_ideal_for_noise + testing(
        networkByNoise, inp)

    confMatr_noise_for_noise = confMatr_noise_for_noise + testing(
        networkByNoise, inp_noise_test)

sc = 72
x = [chr(i) for i in range(sc, sc + 10)]

all_conf_matr = [
    confMatr_ideal_for_ideal / np.amax(confMatr_ideal_for_ideal),
    confMatr_noise_for_ideal / np.amax(confMatr_noise_for_ideal),
    confMatr_ideal_for_noise / np.amax(confMatr_ideal_for_noise),
    confMatr_noise_for_noise / np.amax(confMatr_noise_for_noise)]

y = [[], [], [], []]
for k in range(4):
    for i in range(Nsymbols):
        y[k].append(all_conf_matr[k][i][i])

plt.figure(figsize=(8.4, 6))

plt.subplot(2, 2, 1)
plt.bar(x, y[0])
plt.title('Идеальная НН по идеальным данным')
plt.subplot(2, 2, 3)
plt.bar(x, y[1])
plt.title('Идеальная НН по шумным данным')
plt.subplots_adjust(wspace=0.25, hspace=0.35)
plt.subplot(2, 2, 2)
plt.bar(x, y[2])
plt.title('Шумовая НН по идеальным данным')
plt.subplot(2, 2, 4)
plt.bar(x, y[3])
plt.title('Шумовая НН по шумным данным')
plt.subplots_adjust(wspace=0.25, hspace=0.35)
plt.savefig('Hist_NN.jpg')