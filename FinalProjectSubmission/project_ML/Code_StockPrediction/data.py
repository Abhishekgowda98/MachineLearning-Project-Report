import csv
import numpy as np
from numpy import genfromtxt
from subprocess import call

def shuffle_in_unison(a, b, state):
    #randomly shuffles a and b arrays.
    rng_state = state
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b

def load_ICICI():
    #data = genfromtxt("C:\CMPSCI_589\Project\Data\\ICICI_train.csv", delimiter=',')
    data = genfromtxt("..\Data\ICICI_train.csv", delimiter=',')
    data_X = data[1:data.shape[0], 1:-1]
    data_Y = data[1:data.shape[0], -1]
    np.random.seed(24)
    state = np.random.get_state()
    train_x, train_y = shuffle_in_unison(data_X, data_Y, state)

    test_data = genfromtxt("..\Data\ICICI_test.csv", delimiter=',')
    test_data_X = test_data[1:test_data.shape[0], 1:-1]
    test_data_Y = test_data[1:test_data.shape[0], -1]
    state = np.random.get_state()
    #test_X, test_Y = shuffle_in_unison(test_data_X, test_data_Y, state)
    return train_x, train_y, test_data_X, test_data_Y

def load_TATA():
    data = genfromtxt("..\Data\TATA_train.csv", delimiter=',')
    data_X = data[1:data.shape[0], 1:-1]
    data_Y = data[1:data.shape[0], -1]
    np.random.seed(24)
    state = np.random.get_state()
    train_x, train_y = shuffle_in_unison(data_X, data_Y, state)

    test_data = genfromtxt("..\Data\TATA_test.csv", delimiter=',')
    test_data_X = test_data[1:test_data.shape[0], 1:-1]
    test_data_Y = test_data[1:test_data.shape[0], -1]
    state = np.random.get_state()
    test_X, test_Y = shuffle_in_unison(test_data_X, test_data_Y, state)
    return train_x, train_y, test_X, test_Y

def load_VEDL():
    data = genfromtxt("..\Data\VEDL_train.csv", delimiter=',')
    data_X = data[1:data.shape[0], 1:-1]
    data_Y = data[1:data.shape[0], -1]
    np.random.seed(24)
    state = np.random.get_state()
    train_x, train_y = shuffle_in_unison(data_X, data_Y, state)

    test_data = genfromtxt("..\Data\VEDL_test.csv", delimiter=',')
    test_data_X = test_data[1:test_data.shape[0], 1:-1]
    test_data_Y = test_data[1:test_data.shape[0], -1]
    state = np.random.get_state()
    test_X, test_Y = shuffle_in_unison(test_data_X, test_data_Y, state)
    return train_x, train_y, test_X, test_Y

def load_REDDY():
    data = genfromtxt("..\Data\RDY_train.csv", delimiter=',')
    data_X = data[1:data.shape[0], 1:-1]
    data_Y = data[1:data.shape[0], -1]
    np.random.seed(24)
    state = np.random.get_state()
    train_x, train_y = shuffle_in_unison(data_X, data_Y, state)

    test_data = genfromtxt("..\Data\RDY_test.csv", delimiter=',')
    test_data_X = test_data[1:test_data.shape[0], 1:-1]
    test_data_Y = test_data[1:test_data.shape[0], -1]
    state = np.random.get_state()
    test_X, test_Y = shuffle_in_unison(test_data_X, test_data_Y, state)
    return train_x, train_y, test_X, test_Y