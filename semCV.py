from elmUtil import hidden_matrix, test

import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

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
accNeuronSCV = []

np.random.seed(0)
# variação de neuronios na camada oculta
for b in range(1, 400, 1): #1797 amostras
    # com cross validation
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

    # sem cross-validation

    x_train, x_test, y_train, y_test = train_test_split(digits.data, y, test_size=0.3)
    # número de neuronios
    hidden_n = b

    # treino
    input_size = x_train.shape[1]
    weight_input = np.random.normal(size=[input_size, hidden_n])
    H = hidden_matrix(x_train, weight_input)
    Ht = np.transpose(H)
    # mínimo quadrado B = (H'H)^-1 H'y
    weight_output = np.dot(np.linalg.pinv(np.dot(Ht, H)), np.dot(Ht, y_train))

    # predição
    x = hidden_matrix(x_test, weight_input)
    pred = np.dot(x, weight_output)
    # teste
    accNeuronSCV.append(test(pred.shape[0], pred, y_test))


# plot
pplot.style.use('bmh')
# Acurácia para número de neurônios variáveis usando Extreme Learning Machine no dataset Digits
dataframe = pnd.DataFrame({'Com cross-validation':accNeuron, 'Sem cross-validation':accNeuronSCV})
g = dataframe.plot(figsize=(7, 5), linewidth=0.6)
g.set_xlabel("Número de neurônios na camada oculta")
g.set_ylabel("Acurácia")
pplot.show()