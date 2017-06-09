#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:14:32 2017

@author: sanketh
"""

#stock project

import pandas as pd
import matplotlib.pyplot as plt
#Please set path here
path = './DataSets/'


#Plots the maximum close value for a range
def get_max_close(symbol):
    print "here",symbol
    df = pd.read_csv("../DataSets/{symbol}".format(symbol=symbol))
    df = df.iloc[::-1]
    df[['Close']].plot()
    plt.show()
    return df['Close'].max()
    
#Plots the normalized values of two stocks
def normalized_plot(df1):
    df1 = (df1/df1.ix[0,:])[['NYSE_TATAMOTORS_LIMITED_TTM.csv' + 'Close', 'NSE_TATAMOTORS.NS.csv' + 'Close']]
    df1.columns = ['Tata Motors NYSE prices', 'Tata Motors NSE prices']
    df1.plot(title = "Normalized Tata Motors prices on NYSE and NSE")
    plt.savefig("Figures/normalized_plot_Tata.pdf")
    plt.show()    

#function for data frame operations for pre-processing    
def final_run():
    start_date = '2012-03-23'
    end_date = '2017-01-23'
    dates = pd.date_range(start_date, end_date)
    df1 = pd.DataFrame(index = dates)
    for filename in ['NYSE_TATAMOTORS_LIMITED_TTM.csv', 'NSE_TATAMOTORS.NS.csv']:#glob.glob(os.path.join(path, '*.csv')):
        print "Max close",
        print filename, get_max_close(filename)
        temp_df = pd.read_csv('./DataSets/{}'.format(filename), index_col = "Date", parse_dates = True, usecols = ['Date','Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], na_values = ['nan'])
        temp_df = temp_df.rename(columns = {'Adj Close': filename + 'Adj Close', 'Open' : filename + 'Open', 'High' : filename + 'High', 'Low' : filename + 'Low', 'Close' : filename + 'Close', 'Volume' : filename + 'Volume'})
        df1 = df1.join(temp_df)
    tempdf = pd.read_csv('./DataSets/shuffled_predicted_Tata.csv', index_col = "Date", parse_dates = True, usecols = ['Date', 'Close'], na_values = ['nan'])
    ax = df1['NSE_TATAMOTORS.NS.csvClose'].plot(title = "Close rolling mean", label = 'Close')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc = 'best')
    plt.show()
    #removes non values
    df1 = df1.dropna()
    #creates rolling mean and daily returns features
    df1['rolling mean NSEClose'] = df1['NSE_TATAMOTORS.NS.csvClose'].rolling(5).mean()
    df1['rolling mean NYSEClose'] = df1['NYSE_TATAMOTORS_LIMITED_TTM.csvClose'].rolling(5).mean()
    df1['daily return NYSE'] = df1['NYSE_TATAMOTORS_LIMITED_TTM.csvClose'].div(df1['NYSE_TATAMOTORS_LIMITED_TTM.csvClose'].shift(1))-1
    df1['daily return NSE'] = df1['NSE_TATAMOTORS.NS.csvClose'].div(df1['NSE_TATAMOTORS.NS.csvClose'].shift(1))-1

    df1 = df1.join(tempdf)
    df1.Close = df1.Close.shift(-1)
    df1 = df1.dropna()
    df1.to_csv('test_Tata.csv', encoding = 'utf-8')
    normalized_plot(df1)
        
        
if __name__ == "__main__":
    final_run()
    pass
        