import tensorflow as tf
from sklearn import preprocessing, model_selection
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

name = 'classes.csv'
data = pd.read_csv(name)
data.tail()
data.dropna()

le = LabelEncoder()
data['Spectral'] = le.fit_transform(data['Spectral Class'])
data['Clr'] = le.fit_transform(data['Star color'])

x = np.array(data.drop(['Star type', 'Star color', 'Spectral Class'], 1))
y = np.array(data['Spectral'], dtype='float')
y.shape = (len(y), 1)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,
                                                                    train_size=0.8)

x_f_train = preprocessing.scale(x_train)
x_f_test = preprocessing.scale(x_test)
y_f_train = y_train
y_f_test = y_test

mx = 0
n = 10
counter = 0
lst = []
for i in range(n):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(11, activation=tf.nn.relu))

    model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))

    model.compile(optimizer='ADAM',  # ADAM
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_f_train, y_f_train, epochs=2100, batch_size=2)
    score, acc = model.evaluate(x_f_test, y_f_test)
    if acc > mx:
        mx = acc
        if mx == 1:
            counter += 1
            lst.append(i)
print('Максимум за', n, 'прогонов:', mx, 'Единиц было раз:', counter,
      'Номера таких запусков:', lst)


def doobuch():
    model.fit(x_f_train, y_f_train, epochs=2100, batch_size=2)
    score, acc = model.evaluate(x_f_test, y_f_test)
