from elmUtil import hidden_matrix, hidden_matrix_sigmoid, hidden_matrix_tanh

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
MAEmean2 = 0
MAEmean3 = 0
# mae do número de neurônios atual
MAEneuron = []
MAEneuron2 = []
MAEneuron3 = []

np.random.seed(2)
# variação de neuronios na camada oculta
for b in range(1, 400, 1): #506 amostras
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
        H2 = hidden_matrix_sigmoid(x_trainb, weight_input)
        H2t = np.transpose(H2)
        H3 = hidden_matrix_tanh(x_trainb, weight_input)
        H3t = np.transpose(H3)


        # mínimo quadrado B = (H'H)^-1 H'y
        weight_output = np.dot(np.linalg.pinv(np.dot(Ht, H)), np.dot(Ht, y_train))
        weight_output2 = np.dot(np.linalg.pinv(np.dot(H2t, H2)), np.dot(H2t, y_train))
        weight_output3 = np.dot(np.linalg.pinv(np.dot(H3t, H3)), np.dot(H3t, y_train))

        # predição
        x_testb = np.column_stack([x_test, np.ones([x_test.shape[0], 1])])

        x = hidden_matrix(x_testb, weight_input)
        x2 = hidden_matrix_sigmoid(x_testb, weight_input)
        x3 = hidden_matrix_tanh(x_testb, weight_input)

        pred = np.dot(x, weight_output)
        pred2 = np.dot(x2, weight_output2)
        pred3 = np.dot(x3, weight_output3)

        # teste
        MAEmean += mean_absolute_error(y_test, pred)
        MAEmean2 += mean_absolute_error(y_test, pred2)
        MAEmean3 += mean_absolute_error(y_test, pred3)

    MAEneuron.append(MAEmean / 10)
    MAEneuron2.append(MAEmean2 / 10)
    MAEneuron3.append(MAEmean3 / 10)
    MAEmean = 0
    MAEmean2 = 0
    MAEmean3 = 0

# plot
pplot.style.use('bmh')
# MAE para número de neurônios variáveis usando Extreme Learning Machine no dataset Boston
dataframe = pnd.DataFrame({'ReLU':MAEneuron,'Sigmoid':MAEneuron2, 'Tanh':MAEneuron3})
g = dataframe.plot(figsize=(7, 5), linewidth=0.6)
g.set_xlabel("Número de neurônios na camada oculta")
g.set_ylabel("Mean Absolute Error")
pplot.show()

