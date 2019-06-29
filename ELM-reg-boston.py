from elmUtil import hidden_matrix, reg_neuron_numbers

import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import pandas as pnd
import matplotlib.pyplot as pplot

# dataset
boston = load_boston()

# tamanho da divisao de cross validation
fold = KFold(n_splits=10, shuffle=True)

# mean absolute error médio
MAEmean = 0
# mae do número de neurônios atual
MAEneuron = []

np.random.seed(0)
# variação de neuronios na camada oculta
for b in range(1, 450, 1): #506 amostras
    # cross validation
    for train_index, test_index in fold.split(boston.data, boston.target):
        x_train, x_test = boston.data[train_index], boston.data[test_index]
        y_train, y_test = boston.target[train_index], boston.target[test_index]

        # Standard Scaler
        # entradas
        std_features = StandardScaler()
        x_train = std_features.fit_transform(x_train)
        x_test = std_features.transform(x_test)

        # saídas
        std_out = StandardScaler()
        y_train = std_out.fit_transform(np.expand_dims(y_train, 1))
        Y = max(abs(y_train))
        y_train = y_train / Y
        y_test = std_out.transform(np.expand_dims(y_test, 1))
        y_test = y_test / Y

        # número de neuronios
        hidden_n = b

        # treino
        x_trainb = np.column_stack([x_train, np.ones([x_train.shape[0], 1])])
        input_size = x_trainb.shape[1]
        weight_input = np.random.normal(size=[input_size, hidden_n])
        H = hidden_matrix(x_trainb, weight_input)
        Ht = np.transpose(H)
        # mínimo quadrado B = (H'H)^-1 H'y
        weight_output = np.dot(np.linalg.pinv(np.dot(Ht, H)), np.dot(Ht, y_train))

        # predição
        x_testb = np.column_stack([x_test, np.ones([x_test.shape[0], 1])])
        x = hidden_matrix(x_testb, weight_input)
        pred = np.dot(x, weight_output)

        # teste
        MAEmean += mean_absolute_error(y_test, pred)
        # mean_squared_error() pra mse

    MAEneuron.append(MAEmean / 10)
    MAEmean = 0

reg_neuron_numbers(MAEneuron)

# plot
pplot.style.use('bmh')
# MAE para número de neurônios variáveis usando Extreme Learning Machine no dataset Boston
dataframe = pnd.DataFrame({'':MAEneuron})
g = dataframe.plot(figsize=(7, 5), legend=False, linewidth=0.6)
g.set_xlabel("Número de neurônios na camada oculta")
g.set_ylabel("Mean Absolute Error")
pplot.show()

