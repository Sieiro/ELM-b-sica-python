from elmUtil import hidden_matrix, test, class_neuron_numbers

import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold

import pandas as pnd
import matplotlib.pyplot as pplot

# dataset
digits = load_digits()

# cria matriz one hot com 10 saídas
y = np.zeros([digits.target.shape[0], 10])
for i in range(digits.target.shape[0]):
    y[i][digits.target[i]] = 1

# tamanho da divisao de cross validation
fold = StratifiedKFold(n_splits=10, shuffle=True)

# acurácia média
accMean = 0
# acurácia do número de  neurônios atual
accNeuron = []
# acurácia máxima entre os números de neurônios
accMax = 0
# número de neurônios correspondente a acurácia máxima
accMaxN = 0

np.random.seed(0)
# variação de neuronios na camada oculta
for b in range(1, 800, 1):#1797 amostras
    # cross validation
    for train_index, test_index in fold.split(digits.data, digits.target):
        x_train, x_test = digits.data[train_index], digits.data[test_index]
        y_train, y_test = y[train_index], y[test_index]

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
        accMean += test(pred.shape[0], pred, y_test)

    accNeuron.append(accMean/10)
    accMean = 0

class_neuron_numbers(accNeuron)

# plot
pplot.style.use('bmh')
# Acurácia para número de neurônios variáveis usando Extreme Learning Machine no dataset Digits
dataframe = pnd.DataFrame({'':accNeuron})
g = dataframe.plot(figsize=(7, 5), legend=False, linewidth=0.6)
g.set_xlabel("Número de neurônios na camada oculta")
g.set_ylabel("Acurácia")
pplot.show()