import numpy as np
from scipy.special import expit

def hidden_matrix(x, weight):
    x = np.dot(x, weight)
    #y = np.maximum(x, 0, x)  # Rectified Linear Unit
    y = x*(x>0)
    return y

def hidden_matrix_sigmoid(x, weight):
    x = np.dot(x, weight)
    y = expit(x) # Sigmoid
    return y

def hidden_matrix_tanh(x, weight):
    x = np.dot(x, weight)
    y = np.tanh(x) # Tangente Hiperbólica
    return y

def test(total, pred, y_test):
    expected = 0
    for i in range(total):
        predicted = np.argmax(pred[i])
        target = np.argmax(y_test[i])
        expected = expected + (1 if predicted == target else 0)
        result = expected/total
    #print('Acurácia: {:f}'.format(result))
    return result


def class_neuron_numbers(accNeuron):
    list = accNeuron.copy()
    acc = []
    neuron = []
    max = 0
    n = 0
    for i in range(3):
        for j in range(len(list)):
            if list[j]>max:
                max = list[j]
                n = j+1
        acc.append(max)
        neuron.append(n)
        list[n-1] = 0
        max = 0

    for i in range(len(acc)):
        print('neurônios: {} , acurácia: {:f}'.format(neuron[i], acc[i]))


def reg_neuron_numbers(accNeuron):
    list = accNeuron.copy()
    mae = []
    neuron = []
    min = 10
    n = 0
    for i in range(3):
        for j in range(len(list)):
            if list[j]<min:
                min = list[j]
                n = j+1
        mae.append(min)
        neuron.append(n)
        list[n-1] = 10
        min = 10

    for i in range(len(mae)):
        print('neurônios: {} , mae: {:f}'.format(neuron[i], mae[i]))