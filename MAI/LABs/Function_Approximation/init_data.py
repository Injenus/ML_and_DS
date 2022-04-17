import math
import torch
import numpy as np
import pandas as pd


def getName(arg):
    gl = globals().copy()
    for i in gl:
        if gl[i] is arg:
            # print(i, gl[i])
            return (i)


def tensor_to_csv(tensor):
    ten_np = tensor.numpy()
    df = pd.DataFrame(ten_np)
    df.to_csv(getName(tensor), index=False)


def from_csv(file_name):
    return torch.from_numpy((pd.read_csv(file_name)).values)


x_train = torch.rand(1500) * 12 * math.pi
y_train = np.cos(x_train) + np.arctan(x_train)
x_test = torch.rand(1500) * 12 * math.pi
x_test.numpy().sort()
y_test = np.cos(x_test) + np.arctan(x_test)
tensor_to_csv(x_train)
tensor_to_csv(y_train)
tensor_to_csv(x_test)
tensor_to_csv(y_test)

