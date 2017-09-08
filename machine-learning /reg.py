import pandas as pd
import quandl
import math

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
print(df.head())
df.dropna(inplace=True)
print(df.tail())
