import numpy as np
import csv
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Perceptron
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import random
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import data
import hyper
from subprocess import call
from sklearn.linear_model import ElasticNet
import Graphs_plotting
import matplotlib.pyplot as plt

def main():

    # #==============================ICICI=================================
    train_X, train_Y, test_X, test_Y = data.load_ICICI()
    clf = LinearRegression()
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    MSE = mean_squared_error(test_Y, predictions)
    RMSE = math.sqrt(MSE)
    print RMSE
    Graphs_plotting.line_graph(500,test_Y, predictions,'Linear Regression Baseline',' ICICI ')

    plt.figure(501, figsize=(6, 4))  # 6x4 is the aspect ratio for the plot
    fit = np.polyfit(test_Y, predictions, 1)
    fit_fn = np.poly1d(fit)
    # fit_fn is now a function which takes in x and returns an estimate for y

    plt.plot(test_Y, predictions, 'yo', test_Y, fit_fn(test_Y), '--k')
    #plt.xlim(0, 500)
    plt.ylim(0, 500)
    plt.ylabel("Predicted Values")  # Y-axis label
    plt.xlabel("Actual Value")  # X-axis label
    plt.title("ICICI_LinearRegression_Baseline_scatterPlot")
    plt.savefig("../Figures/ICICI_LinearRegression_Baseline_scatterPlot_.pdf")
    #plt.show()

    # #==============================TATA=================================
    train_X, train_Y, test_X, test_Y = data.load_TATA()
    clf = LinearRegression()
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    MSE = mean_squared_error(test_Y, predictions)
    RMSE = math.sqrt(MSE)
    print RMSE
    Graphs_plotting.line_graph(600, test_Y, predictions, 'Linear Regression Baseline', ' TATA ')

    plt.figure(601, figsize=(6, 4))  # 6x4 is the aspect ratio for the plot
    fit = np.polyfit(test_Y, predictions, 1)
    fit_fn = np.poly1d(fit)
    # fit_fn is now a function which takes in x and returns an estimate for y

    plt.plot(test_Y, predictions, 'yo', test_Y, fit_fn(test_Y), '--k')
    # plt.xlim(0, 500)
    #plt.ylim(0, 500)
    plt.ylabel("Predicted Values")  # Y-axis label
    plt.xlabel("Actual Value")  # X-axis label
    plt.title("TATA_LinearRegression_Baseline_scatterPlot")
    plt.savefig("../Figures/TATA_LinearRegression_Baseline_scatterPlot_.pdf")
    #plt.show()

    # #==============================VEDL=================================
    train_X, train_Y, test_X, test_Y = data.load_VEDL()
    clf = LinearRegression()
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    MSE = mean_squared_error(test_Y, predictions)
    RMSE = math.sqrt(MSE)
    print RMSE
    Graphs_plotting.line_graph(700, test_Y, predictions, 'Linear Regression Baseline', ' VEDL ')

    plt.figure(701, figsize=(6, 4))  # 6x4 is the aspect ratio for the plot
    fit = np.polyfit(test_Y, predictions, 1)
    fit_fn = np.poly1d(fit)
    # fit_fn is now a function which takes in x and returns an estimate for y

    plt.plot(test_Y, predictions, 'yo', test_Y, fit_fn(test_Y), '--k')
    # plt.xlim(0, 500)
    #plt.ylim(0, 500)
    plt.ylabel("Predicted Values")  # Y-axis label
    plt.xlabel("Actual Value")  # X-axis label
    plt.title("VEDL_LinearRegression_Baseline_scatterPlot")
    plt.savefig("../Figures/VEDL_LinearRegression_Baseline_scatterPlot_.pdf")
    # plt.show()

    # #==============================DR REDDY=================================
    train_X, train_Y, test_X, test_Y = data.load_REDDY()
    clf = LinearRegression()
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    MSE = mean_squared_error(test_Y, predictions)
    RMSE = math.sqrt(MSE)
    print RMSE
    Graphs_plotting.line_graph(700, test_Y, predictions, 'Linear Regression Baseline', ' REDDY ')

    plt.figure(701, figsize=(6, 4))  # 6x4 is the aspect ratio for the plot
    fit = np.polyfit(test_Y, predictions, 1)
    fit_fn = np.poly1d(fit)
    # fit_fn is now a function which takes in x and returns an estimate for y

    plt.plot(test_Y, predictions, 'yo', test_Y, fit_fn(test_Y), '--k')
    # plt.xlim(0, 500)
    #plt.ylim(0, 500)
    plt.ylabel("Predicted Values")  # Y-axis label
    plt.xlabel("Actual Value")  # X-axis label
    plt.title("REDDY_LinearRegression_Baseline_scatterPlot")
    plt.savefig("../Figures/REDDY_LinearRegression_Baseline_scatterPlot_.pdf")
    # plt.show()

if __name__ == "__main__": main()