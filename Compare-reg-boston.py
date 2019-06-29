from elmUtil import hidden_matrix

import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import matplotlib.pyplot as pplot

# dataset
boston = load_boston()

# tamanho da divisao de cross validation
fold = KFold(n_splits=5, shuffle=True)

# Mean Absolute Error
MAEmean = 0
MAEmeanLreg = 0
MAEmeanSVR = 0

np.random.seed(0)
for k in range(100):
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

        # ELM-------------------------------------------------------------------
        # número de neuronios
        hidden_n = 92

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

        # LReg-------------------------------------------------------------------
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        predLREG = lr.predict(x_test)

        # SVM--------------------------------------------------------------------
        svrLinear = SVR(kernel="linear")
        svrLinear.fit(x_train, y_train)
        predSVR = svrLinear.predict(x_test)

    # -----------------------------------------------------------------------
        # teste
        mae = mean_absolute_error(y_test, pred)
        mae_lreg = mean_absolute_error(y_test, predLREG)
        mae_svr = mean_absolute_error(y_test, predSVR)
        MAEmean += mae
        MAEmeanLreg += mae_lreg
        MAEmeanSVR += mae_svr

results = [MAEmean / 500, MAEmeanLreg / 500, MAEmeanSVR / 500]
methods = ["ELM", "Regressão Linear", "SVR"]
for i in range(len(results)):
    print('Método: {} , MAE: {:f}'.format(methods[i], results[i]))

# plot
# Comparação dos erros entre modelos
pplot.figure(figsize=(7, 5))
pplot.bar(range(3), results, align='center')
pplot.xticks(range(3), methods)
pplot.ylabel("Mean Absolute Error")
pplot.ylim([0.06, 0.14])
pplot.show()
