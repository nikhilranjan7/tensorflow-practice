import pandas as pd
import quandl
import math
import time
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

# Decrease features i.e. use only certain important columns
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

# Introducing new columns as Pecentage High_Low can be seen more directly related to the next day prices and other similar features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
#Replace the newly defined empty column with -99999
df.fillna(-99999, inplace=True)

#We want to predict 1% future of the entire length of the dataset. If dataset has 1000 rows, we want next 10 rows
forecast_out = int(math.ceil(0.01*len(df)))

#This column is basically composed of Adj. Close of next 10 days i.e. first element is one 1 pair ahead of others if we have 100 datasets
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())
# features include everything but label column
x = np.array(df.drop(['label'],1))
y = np.array(df['label'])

#Goal is to get features values between -1 to 1 to increase accuracy
x = preprocessing.scale(x)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)


# Simple linear Regression
a = time.time()
clf = LinearRegression()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)
b = time.time()
print("LinearRegression took",round(b-a,3),"seconds")

# SVM Regression
a = time.time()
clf = svm.SVR()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)
b = time.time()
print("SVM took",round(b-a,3),"seconds")
