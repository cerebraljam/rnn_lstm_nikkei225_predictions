# Using LSTM to predict the Tokyo Stock Exchange variations

One of the first exercise we are proposed to do with Long Short Term Memory models is to try to predict the stock exchange variations.

When we look at the graphs produced by these models, it's quite impressive to see the Training and Test line hug the real price so close, but how good are these predictions?

Hypothesis:
* We only try to predict the next day, because more than that, any model just have no clue where things are going, and the accumulation of unknown just makes it even less reliable. So day traiding scenario only.
* Because values are normalized in order to train the model, the prediction will also be normalized, which means that it will not give a price. We could try to convert the value back to a real value, but is it really necessary? If the objective is just to guess which stock will go up, and how confident we are that it will go up, just a general indication (up/down) could be sufficient.

With these hypothesis, I tried to train a model that would look at the past values, and try to guess which stock should be purchased tomorrow morning, because it should be going up during the day.

## Conclusion

See at the end.

## Spoilers

It's quite unreliable. I wouldn't use this to blindly chose my investments.

## Importing libraries


```python
import matplotlib.pyplot as plt
# import seaborn
import json
from datetime import date
import datetime
import random
import time
from os import path
import pandas as pd
import numpy as np
from scipy.stats import beta
from math import sqrt

import requests

# from pandas_datareader import data as dr

from html.parser import HTMLParser
from html.entities import name2codepoint

# pip install opencv-python matplotlib numpy scipy keras scikit-learn tensorflow

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

np.random.seed(42)

# pip install yfinance
import yfinance as yf

%matplotlib inline

# %load_ext autoreload
# %autoreload 2

```

    Using TensorFlow backend.


# Load/Update of the list of symbols from Nikkei 225 site

This next cell checks if there is a nikkei225.json file in the current folder and loads it. If there isn't one, it will go recover the latest list on the official nikkei site. 

(Source of the Nikkei 225 companies: https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225. There is a screw up on their site for Osaka Gas, and I am too lazy to fix it in the code. There is one company that doesn't display well, and it's that one.)


```python
nfile = 'nikkei225.json'

class MyHTMLParser(HTMLParser):    
    def handle_starttag(self, tag, attrs):
        global dept, nikkei225, area, company, location
        
        dept += 1
        for attr in attrs:
            if attr[0] == 'class':
                if attr[1] == 'col-xs-11 col-sm-11' and dept == 8:
                    location = 0
                    company = {
                        'code': "",
                        'url': "",
                        'name': "",
                        'area': ""
                    }
                elif attr[1] == 'col-xs-3 col-sm-1_5' and dept == 8:
                    location = 1
            elif attr[0] == 'href' and location == 2:
                company['url'] = attr[1]
                location = 3

    def handle_endtag(self, tag):
        global dept
        dept -= 1

    def handle_data(self, data):
        global dept, nikkei225, area, company, location
        if location == 0 and dept == 8:
            print("Area:", data)
            company["area"] = data
            location = None
        elif location == 1 and dept == 8 and data not in ['Code', 'Company Name']:
            company['code'] = data + ".T"
            location = 2
        elif location == 3 and dept == 8:
            company['name'] = data
            nikkei225[company['code']] = dict(company)                

            
if path.exists(nfile):
    if (time.time() - path.getmtime(nfile) < (7 * 24 * 60 * 60)): # Update the list every week
        print('the current {} file is good enough'.format(nfile))
        with open(nfile, 'r') as infile:
            nikkei225 = json.load(infile)
    else:
        print('the current {} file is too old'.format(nfile))
        nikkei225 = {}
        
if len(nikkei225.keys()) == 0:
    nikkei225 = {}
    company = {}
    area = ""
    dept = 0
    location = None
    
    r = requests.get('https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225')
    print('http get status', r.status_code, 'length', len(r.text))
    parser = MyHTMLParser()
    parser.feed(r.text)
    
    with open(nfile, 'w') as outfile:
        json.dump(nikkei225, outfile)

print('Number of symbols in the Nikkei 225 index: {}'.format(len(nikkei225.keys())))
```

    the current nikkei225.json file is too old
    http get status 200 length 76143
    Area: Pharmaceuticals
    Area: Electric Machinery
    Area: Automobiles & Auto parts
    Area: Precision Instruments
    Area: Communications
    Area: Banking
    Area: Other Financial Services
    Area: Securities
    Area: Insurance
    Area: Fishery
    Area: Foods
    Area: Retail
    Area: Services
    Area: Mining
    Area: Textiles & Apparel
    Area: Pulp & Paper
    Area: Chemicals
    Area: Petroleum
    Area: Rubber
    Area: Glass & Ceramics
    Area: Steel
    Area: Nonferrous Metals
    Area: Trading Companies
    Area: Construction
    Area: Machinery
    Area: Shipbuilding
    Area: Other Manufacturing
    Area: Real Estate
    Area: Railway & Bus
    Area: Land Transport
    Area: Marine Transport
    Area: Air Transport
    Area: Warehousing
    Area: Electric Power
    Area: Gas
    Number of symbols in the Nikkei 225 index: 225


# Download the current symbol values from the Yahoo Finance Site


```python
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)
print('today', today.strftime("%Y-%m-%d"), 'tomorrow', tomorrow.strftime("%Y-%m-%d"))

tickers = list(nikkei225.keys())
start_date = '2019-01-01'
end_date = tomorrow
```

    today 2020-05-14 tomorrow 2020-05-15



```python
# Downloading the values from Yahoo Finance here
df_raw_nikkei225 = yf.download(tickers, start=start_date, end=end_date)
```

    [*********************100%***********************]  225 of 225 completed



```python
# A bit of cleanup...
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
df_clean_nikkei2225 = df_raw_nikkei225.reindex(all_weekdays)
df_clean_nikkei2225 = df_clean_nikkei2225.fillna(method='ffill')
df_clean_nikkei2225.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Adj Close</th>
      <th>...</th>
      <th colspan="10" halign="left">Volume</th>
    </tr>
    <tr>
      <th></th>
      <th>1332.T</th>
      <th>1333.T</th>
      <th>1605.T</th>
      <th>1721.T</th>
      <th>1801.T</th>
      <th>1802.T</th>
      <th>1803.T</th>
      <th>1808.T</th>
      <th>1812.T</th>
      <th>1925.T</th>
      <th>...</th>
      <th>9502.T</th>
      <th>9503.T</th>
      <th>9531.T</th>
      <th>9532.T</th>
      <th>9602.T</th>
      <th>9613.T</th>
      <th>9735.T</th>
      <th>9766.T</th>
      <th>9983.T</th>
      <th>9984.T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-11</th>
      <td>479.0</td>
      <td>2317.0</td>
      <td>697.000000</td>
      <td>2992.0</td>
      <td>3535.0</td>
      <td>952.0</td>
      <td>875.0</td>
      <td>1208.0</td>
      <td>1167.0</td>
      <td>2778.5</td>
      <td>...</td>
      <td>1655000.0</td>
      <td>2343500.0</td>
      <td>1093400.0</td>
      <td>1066300.0</td>
      <td>410800.0</td>
      <td>4610500.0</td>
      <td>373400.0</td>
      <td>677000.0</td>
      <td>1189900.0</td>
      <td>11934200.0</td>
    </tr>
    <tr>
      <th>2020-05-12</th>
      <td>476.0</td>
      <td>2319.0</td>
      <td>690.299988</td>
      <td>2966.0</td>
      <td>3520.0</td>
      <td>946.0</td>
      <td>864.0</td>
      <td>1195.0</td>
      <td>1160.0</td>
      <td>2731.5</td>
      <td>...</td>
      <td>1270100.0</td>
      <td>1991200.0</td>
      <td>900100.0</td>
      <td>1543500.0</td>
      <td>330200.0</td>
      <td>3603000.0</td>
      <td>524800.0</td>
      <td>436900.0</td>
      <td>1042600.0</td>
      <td>12076700.0</td>
    </tr>
    <tr>
      <th>2020-05-13</th>
      <td>473.0</td>
      <td>2307.0</td>
      <td>682.799988</td>
      <td>3010.0</td>
      <td>3475.0</td>
      <td>940.0</td>
      <td>860.0</td>
      <td>1194.0</td>
      <td>1145.0</td>
      <td>2744.0</td>
      <td>...</td>
      <td>1345100.0</td>
      <td>4470700.0</td>
      <td>1030700.0</td>
      <td>1377200.0</td>
      <td>393400.0</td>
      <td>3617900.0</td>
      <td>777500.0</td>
      <td>653200.0</td>
      <td>1193300.0</td>
      <td>18234300.0</td>
    </tr>
    <tr>
      <th>2020-05-14</th>
      <td>462.0</td>
      <td>2305.0</td>
      <td>663.400024</td>
      <td>3005.0</td>
      <td>3485.0</td>
      <td>918.0</td>
      <td>846.0</td>
      <td>1157.0</td>
      <td>1144.0</td>
      <td>2450.5</td>
      <td>...</td>
      <td>1312800.0</td>
      <td>3607100.0</td>
      <td>817200.0</td>
      <td>987400.0</td>
      <td>329200.0</td>
      <td>4056100.0</td>
      <td>669400.0</td>
      <td>780100.0</td>
      <td>755200.0</td>
      <td>17091200.0</td>
    </tr>
    <tr>
      <th>2020-05-15</th>
      <td>462.0</td>
      <td>2305.0</td>
      <td>663.400024</td>
      <td>3005.0</td>
      <td>3485.0</td>
      <td>918.0</td>
      <td>846.0</td>
      <td>1157.0</td>
      <td>1144.0</td>
      <td>2450.5</td>
      <td>...</td>
      <td>1312800.0</td>
      <td>3607100.0</td>
      <td>817200.0</td>
      <td>987400.0</td>
      <td>329200.0</td>
      <td>4056100.0</td>
      <td>669400.0</td>
      <td>780100.0</td>
      <td>755200.0</td>
      <td>17091200.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1350 columns</p>
</div>




```python
# keeping a dataframe with just the close values
df_close_nikkei225 = df_clean_nikkei2225['Close']

# here I am adding few values, mainly rolling averages over 5,20 and 100 days, 
# then the ratio between the 5 and 20 days rolling average, then 20 and 100 days
for symbol in [x for x in nikkei225.keys()]:
    df_clean_nikkei2225[('RollAvg5', symbol)] = df_close_nikkei225.loc[:, symbol].rolling(window=5).mean()
    df_clean_nikkei2225[('RollAvg20', symbol)] = df_close_nikkei225.loc[:, symbol].rolling(window=20).mean()
    df_clean_nikkei2225[('RollAvg100', symbol)] = df_close_nikkei225.loc[:, symbol].rolling(window=100).mean()
    df_clean_nikkei2225[('ratio5-20', symbol)] = df_clean_nikkei2225[('RollAvg5', symbol)] / df_clean_nikkei2225[('RollAvg20', symbol)]
    df_clean_nikkei2225[('ratio20-100', symbol)] = df_clean_nikkei2225[('RollAvg20', symbol)] / df_clean_nikkei2225[('RollAvg100', symbol)]
    

# keeping a dataframe with the relevant values used for our model
df_relev_nikkei225 = df_clean_nikkei2225[['Close', 'High', 'Low', 'Volume', 'RollAvg5', 'RollAvg20', 'RollAvg100', 'ratio5-20', 'ratio20-100']]
df_relev_nikkei225.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Close</th>
      <th>...</th>
      <th colspan="10" halign="left">ratio20-100</th>
    </tr>
    <tr>
      <th></th>
      <th>1332.T</th>
      <th>1333.T</th>
      <th>1605.T</th>
      <th>1721.T</th>
      <th>1801.T</th>
      <th>1802.T</th>
      <th>1803.T</th>
      <th>1808.T</th>
      <th>1812.T</th>
      <th>1925.T</th>
      <th>...</th>
      <th>9101.T</th>
      <th>9104.T</th>
      <th>9107.T</th>
      <th>9202.T</th>
      <th>9301.T</th>
      <th>9501.T</th>
      <th>9502.T</th>
      <th>9503.T</th>
      <th>9531.T</th>
      <th>9532.T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-11</th>
      <td>479.0</td>
      <td>2317.0</td>
      <td>697.000000</td>
      <td>2992.0</td>
      <td>3535.0</td>
      <td>952.0</td>
      <td>875.0</td>
      <td>1208.0</td>
      <td>1167.0</td>
      <td>2778.5</td>
      <td>...</td>
      <td>0.856299</td>
      <td>0.788055</td>
      <td>0.788715</td>
      <td>0.782871</td>
      <td>0.910213</td>
      <td>0.903200</td>
      <td>1.009816</td>
      <td>0.949279</td>
      <td>1.022829</td>
      <td>1.065387</td>
    </tr>
    <tr>
      <th>2020-05-12</th>
      <td>476.0</td>
      <td>2319.0</td>
      <td>690.299988</td>
      <td>2966.0</td>
      <td>3520.0</td>
      <td>946.0</td>
      <td>864.0</td>
      <td>1195.0</td>
      <td>1160.0</td>
      <td>2731.5</td>
      <td>...</td>
      <td>0.860282</td>
      <td>0.792733</td>
      <td>0.797240</td>
      <td>0.785206</td>
      <td>0.915525</td>
      <td>0.903117</td>
      <td>1.008801</td>
      <td>0.945991</td>
      <td>1.021799</td>
      <td>1.066899</td>
    </tr>
    <tr>
      <th>2020-05-13</th>
      <td>473.0</td>
      <td>2307.0</td>
      <td>682.799988</td>
      <td>3010.0</td>
      <td>3475.0</td>
      <td>940.0</td>
      <td>860.0</td>
      <td>1194.0</td>
      <td>1145.0</td>
      <td>2744.0</td>
      <td>...</td>
      <td>0.866637</td>
      <td>0.797843</td>
      <td>0.805641</td>
      <td>0.786181</td>
      <td>0.922060</td>
      <td>0.902469</td>
      <td>1.007270</td>
      <td>0.941941</td>
      <td>1.019614</td>
      <td>1.067589</td>
    </tr>
    <tr>
      <th>2020-05-14</th>
      <td>462.0</td>
      <td>2305.0</td>
      <td>663.400024</td>
      <td>3005.0</td>
      <td>3485.0</td>
      <td>918.0</td>
      <td>846.0</td>
      <td>1157.0</td>
      <td>1144.0</td>
      <td>2450.5</td>
      <td>...</td>
      <td>0.873170</td>
      <td>0.803155</td>
      <td>0.813659</td>
      <td>0.787633</td>
      <td>0.928259</td>
      <td>0.901396</td>
      <td>1.004216</td>
      <td>0.936696</td>
      <td>1.017186</td>
      <td>1.066811</td>
    </tr>
    <tr>
      <th>2020-05-15</th>
      <td>462.0</td>
      <td>2305.0</td>
      <td>663.400024</td>
      <td>3005.0</td>
      <td>3485.0</td>
      <td>918.0</td>
      <td>846.0</td>
      <td>1157.0</td>
      <td>1144.0</td>
      <td>2450.5</td>
      <td>...</td>
      <td>0.877798</td>
      <td>0.806760</td>
      <td>0.820214</td>
      <td>0.788366</td>
      <td>0.932850</td>
      <td>0.900180</td>
      <td>1.001429</td>
      <td>0.931259</td>
      <td>1.014700</td>
      <td>1.065853</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2025 columns</p>
</div>



## Graphing a single symbol with rolling averages (5, 20, 50 days)


```python
## This cell is just for testing. it will graph a symbol after recalculating rolling averages
# The graph_symbol function is also used later on.

def graph_symbol(symbol, df):
    print(symbol)
    # Get the timeseries. This now returns a Pandas Series object indexed by date.

    wtv = df.loc[:, symbol]

    roll_20_days = wtv.rolling(window=20).mean()
    roll_100_days = wtv.rolling(window=100).mean()

    # Plot everything
    fig, ax = plt.subplots(figsize=(16,3))

    ax.plot(wtv.index, wtv, label=symbol)
    ax.plot(roll_20_days.index, roll_20_days, label='20 days rolling')
    ax.plot(roll_100_days.index, roll_100_days, label='100 days rolling')

    ax.set_xlabel('Date')
    ax.set_ylabel('Adjusted closing price (¥)')
    ax.legend()
    plt.show()

graph_symbol('4755.T', df_close_nikkei225)
```

    4755.T



![png](tse_lstm_predictions_files/tse_lstm_predictions_11_1.png)


## Defining the model

The next cell defines the function that will return a LSTN RNN.


```python
# LSTM Related functions

def build_part1_RNN(window_size, n_features = 1, nodes = 5, dropout=0.2):
    model = Sequential()
    model.add(LSTM(nodes, activation='tanh', input_shape=(window_size, n_features), dropout=dropout))
    model.add(Dense(1))
    
    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    # model.compile(loss='mae', optimizer='adam')
        
    return model
```

## Preparing the data to train the model

The `window_transform_3d_series` function returns the series days (X) and the value that need to be predicted (y)


```python
# convert series to supervised learning
def window_transform_3D_series(df, window_size, out=['Close']):
    # containers for input/output pairs
    X = []
    y = []
    
    for ii in range(len(df) - window_size):
        X.append(list(df.iloc[ii:ii+window_size].values))
        y.append(df.iloc[ii+window_size][out].values)

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:3])
    y = np.asarray(y)
    y.shape = (len(y),1)
    

    return X, y

def mean_beta(a, b):
    mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
    return mean

if False: #testing function, just to confirm that the values are properly returned by window_transform_3D_series
    window_size = 8
    features = ['Close']
    n_features = len(features)

    for symbol in list(df_close_nikkei225.keys())[:1]:
        print(symbol)
        df = df_relev_nikkei225.loc[:, [(x, symbol) for x in features]]

        df_scaled=(df-df.mean())/df.std()

        X,y = window_transform_3D_series(df_scaled.fillna(0).head(20),window_size, 'Close')
        print([{"X": list(X[i][-1]),"y": y[i][0]} for i in range(len(y))])
```

## Initializing the model


```python
def training(nikkei225, df_relev_nikkei225, model, window_size, features, epochs, batch_size, verbose=True):
    n_features = len(features)
    
    # if set to True, training will be done on all the nikkei225 companies. However, not all companies behave 
    # the same way while I coded this, most probably because of the pandemic. 
    # Because of the virus and the potential for high profit through speculations,
    # I decided to focus the model on pharmaceutical area companies
    
    if False:
        array = [x for x in nikkei225.keys()]
    else:
        array = [x for x in nikkei225.keys() if nikkei225[x]['area'] == 'Pharmaceuticals']
        
    np.random.shuffle(array)
    
    for s in range(len(array)):
        symbol = array[s]
        df = df_relev_nikkei225.loc[:, [(x, symbol) for x in features]]

        df_scaled=(df-df.mean())/df.std()

        X,y = window_transform_3D_series(df_scaled.fillna(0), window_size, 'Close')

        train_test_split = int(np.ceil(4*len(y)/float(5)))   # set the split point

        X_train = X[:train_test_split,:]
        y_train = y[:train_test_split]

        # keep the last chunk for testing
        X_test = X[train_test_split:,:]
        y_test = y[train_test_split:]

        # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
        X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, n_features)))
        X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, n_features)))

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0, shuffle=True)

        # generate predictions for training
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # print out training and testing errors
        training_error = model.evaluate(X_train, y_train, verbose=0)
        testing_error = model.evaluate(X_test, y_test, verbose=0)
        if verbose:
            print('* {} {} training error = {:0.3f} ({}), testing error: {:0.3f} ({})'.format(
                symbol, nikkei225[symbol]['name'], 
                training_error, training_error < error_objective, 
                testing_error, testing_error < error_objective
            ))


    if verbose:
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=[20,15])
        plt.grid(True)
        plt.plot(df_scaled['Close'].reset_index(drop=True),color = 'k')


        # plot training set prediction
        split_pt = train_test_split + window_size 
        plt.plot(np.arange(window_size,split_pt,1),train_predict,color = 'b')

        # plot testing set prediction
        plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')

        # pretty up graph
        plt.xlabel('day')
        plt.ylabel('(normalized) price of {} stock'.format(symbol))
        plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    
    return model
        

```

# Training the model with the Nikkei225 values

Here we are defining the parameters of the model, then we initialize it and start the training.

> Note: The training function is hardcoded to focus only on pharmaceutical companies


```python
# At the end of this of this notebook, there is a function that randomize over multiple parameters
# the objective was to try to identify which values were consitantly providing better predictions.
# Here is a small history of the outputs. All outputs are not consistent because of ongoing evolution in the code

# {'mean': array(0.55892857), 'window_size': 6, 'epochs': 77, 'features': 'Close,ratio5-20,ratio20-100,Volume', 'nodes': 30, 'batch_size': 53, 'dropout': 0.22471143818217415}
# {'mean': array(0.54464286), 'window_size': 6, 'epochs': 66, 'features': 'Close,ratio5-20,ratio20-100', 'nodes': 20, 'batch_size': 51, 'dropout': 0.10690837670124698}
# {'mean': array(0.54285714), 'window_size': 4, 'epochs': 87, 'features': 'Close', 'nodes': 24, 'batch_size': 52, 'dropout': 0.20949785280279093}
# {'mean': array(0.53928571), 'window_size': 4, 'epochs': 52, 'features': ['Close', 'ratio5-20'], 'nodes': 15, 'batch_size': 54, 'dropout': 0.40870996041874336}
# {'mean': array(0.53571429), 'window_size': 5, 'epochs': 99, 'features': ['Close', 'ratio5-20'], 'nodes': 21, 'batch_size': 67, 'dropout': 0.0825648831880042}
# {'mean': array(0.50967742), 'window_size': 5, 'epochs': 59, 'n_features': 3, 'nodes': 17, 'batch_size': 61, 'dropout': 0.15086690591434565}


window_size = 4
features = [
    'Close',
    'Low',
    'High',
    #'RollAvg5', 
    #'RollAvg100',
    'ratio5-20',
    'ratio20-100',
    'Volume',
]
n_features = len(features)

error_objective = 0.02

epochs = 80#50
batch_size = 50
nodes = 20#5 #window_size * 5
dropout = 0.22

model = build_part1_RNN(window_size, n_features, nodes, dropout)

model = training(nikkei225, df_relev_nikkei225, model, window_size, features, epochs, batch_size, True)
```

    * 4578.T OTSUKA HOLDINGS CO., LTD. training error = 0.042 (False), testing error: 0.294 (False)
    * 4151.T KYOWA KIRIN CO., LTD. training error = 0.032 (False), testing error: 0.234 (False)
    * 4523.T EISAI CO., LTD. training error = 0.059 (False), testing error: 0.124 (False)
    * 4519.T CHUGAI PHARMACEUTICAL CO., LTD. training error = 0.022 (False), testing error: 0.262 (False)
    * 4502.T TAKEDA PHARMACEUTICAL CO., LTD. training error = 0.051 (False), testing error: 0.305 (False)
    * 4507.T SHIONOGI & CO., LTD. training error = 0.047 (False), testing error: 0.604 (False)
    * 4506.T SUMITOMO DAINIPPON PHARMA CO., LTD. training error = 0.050 (False), testing error: 0.268 (False)
    * 4503.T ASTELLAS PHARMA INC. training error = 0.058 (False), testing error: 0.278 (False)
    * 4568.T DAIICHI SANKYO CO., LTD. training error = 0.063 (False), testing error: 0.073 (False)



![png](tse_lstm_predictions_files/tse_lstm_predictions_19_1.png)



![png](tse_lstm_predictions_files/tse_lstm_predictions_19_2.png)


Wow! like all the LSTM examples, the prediction really look like the real values! I will be rich! ^_^

# Applying to predict the future

Now what? What can we do with this?

Let's say that we still take all the pharmaceutical companies listed in the Nikkei225, can we guess which ones will go up or down? How successful are the predictions? Can we trust them to help taking a decision about the next day's movements? 

This is what the following cell is doing. 
* It's going through all companies in the pharmaceutical area and rerun the training and testing values against the model. 
* While doing so, it will try to guess if the next day will go up or down, only on the test values, unused during the training.
* and if it goes up, does it goes up significantly enough in order to make profit? (+0.05 of the previously normalized value, this is arbitrary)
* For the one that does not qualify, we just forget about them. They can both be missed opportunities, or cases where value would really had declined.
* We then confirm if we did a good call to `buy` shares.
* At the last step, the code will display which symbol he thinks should be purchased, how many times the model was right (a)/wrong (b), and how much capital we would have if we had followed all the predictions, just for fun.


```python
def best_options(nikkei225, df_relev_nikkei225, model, window_size, features, verbose=False):
    
    df_future = pd.DataFrame(columns = ['date', 'symbol', 'name', 'area', 'close', 'close_normalized', 'prediction', 'direction', 'buy', 'a', 'b', 'mean_beta', 'capital'])
    n_features = len(features)
    aa = 1
    bb = 1
    
    for symbol in sorted([x for x in nikkei225.keys() if nikkei225[x]['area'] == 'Pharmaceuticals']):    
        df = df_relev_nikkei225.loc[:, [(x, symbol) for x in features]]

        mean = df.mean()
        std = df.std()

        df_scaled=(df-mean)/std

        X,y = window_transform_3D_series(df_scaled.fillna(0), window_size, 'Close')

        train_test_split = int(np.ceil(4*len(y)/float(5)))   # set the split point

        # keep the last chunk for testing
        X_test = X[train_test_split:,:]
        y_test = y[train_test_split:]

        # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, n_features] 
        X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, n_features)))

        # generate predictions for training
        test_predict = model.predict(X_test)

        df_scaled_reindexed = df_scaled.reset_index()
        df_reindexed = df.reset_index()

        last_prediction = 4

        a = 1
        b = 1
        capital = 0

        for t in range(len(X_test-1)):

            scaled_sr = df_scaled_reindexed.iloc[train_test_split + t + window_size-1][['index', 'Close']]
            raw_sr = df_reindexed.iloc[train_test_split + t + window_size-1][['index', 'Close']]
            raw_tomorrow_sr = df_reindexed.iloc[train_test_split + t + window_size][['index', 'Close']]

            rday = str(raw_sr['index'].values[0]).split('T')[0]
            rtomorrow = str(raw_tomorrow_sr['index'].values[0]).split('T')[0]

            scaled_close = scaled_sr[('Close', symbol)]
            raw_close = raw_sr[('Close', symbol)]

            predicted = test_predict[t][0]
            decision = False

            if predicted < last_prediction:
                guess = 'Down'.format(predicted)
            else:
                guess = 'Up'.format(predicted)
                if predicted > last_prediction + 0.05:
                    decision = True


            if len(X_test) > t+1:
                scaled_next = X_test[t+1][-1][0]
                raw_next = raw_tomorrow_sr[('Close', symbol)]

                profit =  raw_next - raw_close
                success = False

                if decision:
                    capital += profit
                    if profit > 0:
                        a+=1
                        aa+=1
                    if profit < 0:
                        b+=1
                        bb+=1

                    if False:
                        print('{} close: {} ({:0.3f}), pred: {}, buy: {}, next: {} ({:0.3f}), ¥: {:0.0f} right: {}'.format(
                            rday, raw_close, scaled_close, guess, decision, raw_next, scaled_next, profit, success
                        ))
                if not decision and False: # forgetting about the values that are predicted to go down
                    if profit < 0:
                        a+=1
                        aa+=1
                    if profit > 0:
                        b+=1
                        bb+=1
                    
                    
            else:
                item = {
                    'symbol': symbol,
                    'name': nikkei225[symbol]['name'],
                    'area': nikkei225[symbol]['area'],
                    'date': rday,
                    'close': raw_close,
                    'close_normalized': scaled_close, 
                    'prediction': predicted, 
                    'direction': guess, 
                    'buy': decision, 
                    'a': a, 
                    'b': b, 
                    'mean_beta': mean_beta(a,b),
                    'capital': capital
                }

                df_future = df_future.append(item, ignore_index=True)
                if item['buy']:
                    print("{} {} {} (buy: {}) capital: {}".format(item['symbol'], item['name'], item['area'], item['buy'], capital))

            last_prediction = predicted

    if verbose:
        print(symbol)
        plt.figure(figsize=[20,8])
        plt.grid(True)
        plt.plot(df_scaled['Close'].reset_index(drop=True),color = 'k')

        # plot training set prediction
        split_pt = train_test_split + window_size 

        # plot testing set prediction
        plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')

        # pretty up graph
        plt.xlabel('day')
        plt.ylabel('(normalized) price of {} stock'.format(symbol))
        plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    return mean_beta(aa,bb), df_future
```


```python
mbeta, df_future = best_options(nikkei225, df_relev_nikkei225, model, window_size, features, True)
print('mean_beta: ', mbeta)
```

    4502.T TAKEDA PHARMACEUTICAL CO., LTD. Pharmaceuticals (buy: True) capital: 169.0
    4568.T DAIICHI SANKYO CO., LTD. Pharmaceuticals (buy: True) capital: 1128.0
    4578.T



![png](tse_lstm_predictions_files/tse_lstm_predictions_23_1.png)


    mean_beta:  0.4772727272727273


That last mean_beta value is an important indicator. It means that the model and conditions was successful ~47.7% of the time with the current model and parameters. This seems to be less than a coin flip.

## According to our model, which stock is expected to rise tomorrow?


```python
budget_limit = 2000 # maximum price willing to pay per share

df_future[(df_future['buy'] == True) & (df_future['close'] < budget_limit)].sort_values(['mean_beta'], ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>name</th>
      <th>area</th>
      <th>close</th>
      <th>close_normalized</th>
      <th>prediction</th>
      <th>direction</th>
      <th>buy</th>
      <th>a</th>
      <th>b</th>
      <th>mean_beta</th>
      <th>capital</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



In this case, the model is telling us that we should not buy any share because it believes that one will be profitable enough.


```python
# this cell graphes all the shares identified as potentials by the model.
for i, r in df_future[(df_future['buy'] == True) & (df_future['close'] < budget_limit)].sort_values(['mean_beta'], ascending=False).iterrows():
    graph_symbol(r['symbol'], df_close_nikkei225)
```

## How about the other stocks, what should happen to them?

Ok, the model is telling us that we should not buy anything, but does this mean that no share will be going up? Here, because I did set a maximum budget per share to 2000¥, anything that is more expensive will not be displayed.

The following cell is displaying all the shares that are predicted to go up, regardless of the budget, or the expectations of profits.


```python
df_future[df_future['direction'] == 'Up'].sort_values(['mean_beta'], ascending=False).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>name</th>
      <th>area</th>
      <th>close</th>
      <th>close_normalized</th>
      <th>prediction</th>
      <th>direction</th>
      <th>buy</th>
      <th>a</th>
      <th>b</th>
      <th>mean_beta</th>
      <th>capital</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2020-05-14</td>
      <td>4502.T</td>
      <td>TAKEDA PHARMACEUTICAL CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>4028.0</td>
      <td>-0.011821</td>
      <td>0.482957</td>
      <td>Up</td>
      <td>True</td>
      <td>16</td>
      <td>10</td>
      <td>0.6153846153846154</td>
      <td>169.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-05-14</td>
      <td>4523.T</td>
      <td>EISAI CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>7490.0</td>
      <td>0.211007</td>
      <td>0.203684</td>
      <td>Up</td>
      <td>False</td>
      <td>11</td>
      <td>12</td>
      <td>0.4782608695652174</td>
      <td>-606.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-05-14</td>
      <td>4568.T</td>
      <td>DAIICHI SANKYO CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>8489.0</td>
      <td>1.832643</td>
      <td>1.377511</td>
      <td>Up</td>
      <td>True</td>
      <td>11</td>
      <td>12</td>
      <td>0.4782608695652174</td>
      <td>1128.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-05-14</td>
      <td>4519.T</td>
      <td>CHUGAI PHARMACEUTICAL CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>12970.0</td>
      <td>1.931239</td>
      <td>1.448292</td>
      <td>Up</td>
      <td>False</td>
      <td>9</td>
      <td>11</td>
      <td>0.45</td>
      <td>500.0</td>
    </tr>
  </tbody>
</table>
</div>



### Descriptions of the columns

* date: current date
* symbol: symbol on the Tokyo Stock Exchange (nikkei)
* name: official company name
* area: The nikkei is grouped in multiple areas
* close: closing price
* close_normalized: all values are normalized, this is the value once normalized
* prediction: the model predict that the next close_normalized value should be of x (also normalized)
* direction: here we are comparing against the last prediction value, because often the close value and the prediction value aren't exactly the same
* buy: should we buy or not?
* a: this to calculate the mean beta distribution: a == successfully predicted the next day during training
* b: this to calculate the mean beta distribution: b == failed to predict the next day during training
* mean_beta: mean beta distribution, what is the percentage of success of the model at predicting the model for that symbol
* capital: this is just to toy around. starting with a capital of 0 on 2019-01-01, if we buy when the model tells to buy, how much yens would we have a the end?

## What is the current value of these shares that should be purchased?

I used this sell during my testing. I was doing the simulation the night before, then when it was passed 15:00 JST, I was running this cell to check the new values and see how rich I would have become.


```python
now = datetime.date.today()
yesterday = now - datetime.timedelta(days=1)
tomorrow = now + datetime.timedelta(days=1)
print('yesterday', yesterday.strftime("%Y-%m-%d"), 'now', now.strftime("%Y-%m-%d"), 'tomorrow', tomorrow.strftime("%Y-%m-%d"))

focus_tickers = list(df_future[df_future['direction'] == 'Up']['symbol'].unique())
start_date = yesterday.strftime("%Y-%m-%d")
end_date = tomorrow.strftime("%Y-%m-%d")

# Downloading the values from Yahoo Finance here
df_raw_focus_nikkei225 = yf.download(focus_tickers, start=start_date, end=end_date)

# A bit of cleanup...
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
df_clean_focus_nikkei2225 = df_raw_focus_nikkei225.reindex(all_weekdays)
df_clean_focus_nikkei2225 = df_clean_focus_nikkei2225.fillna(method='ffill')

# keeping a dataframe with just the close values
df_clean_focus_nikkei2225['Close']
```

    yesterday 2020-05-14 now 2020-05-15 tomorrow 2020-05-16
    [*********************100%***********************]  4 of 4 completed





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>4502.T</th>
      <th>4519.T</th>
      <th>4523.T</th>
      <th>4568.T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-14</th>
      <td>4028.0</td>
      <td>12970.0</td>
      <td>7490.0</td>
      <td>8489.0</td>
    </tr>
    <tr>
      <th>2020-05-15</th>
      <td>4012.0</td>
      <td>13105.0</td>
      <td>7454.0</td>
      <td>8432.0</td>
    </tr>
  </tbody>
</table>
</div>



In the 4 symbols that were predicted to go up, 1 did... and 3 didn't.

# Conclusion

The predictions from the LSTM RNN seems to be following the reality, but if we try to use these predictions to take purchasing decisions, we can see that the results look like a coin toss. The number of successes VS failures are quite around 50%, which is consistant with a random guess.

Can we use this in real life? I wouldn't.

# Optimization section

This section is mainly to try multiple parameters over and over, attempting to find the best parameters for the model.

It is currently set not to run.


```python
%%time 

def init_optimization(nikkei225, df_relev_nikkei225, features):
    df_mean_beta = pd.DataFrame(columns=['mean', 'window_size', 'epochs', 'features', 'nodes', 'dropout'])
    
    window_size_range = [3, 6]
    epochs_range = [50, 100]
    nodes_range = [5, 50]
    batch_size_range = [50, 75]
    dropout_range = [0.0, 0.5]
    n_features = len(features)
    
    max_tests = 10
    for r in range(max_tests):
        window_size = random.randint(window_size_range[0], window_size_range[1])
        epochs = np.random.randint(low=epochs_range[0], high=epochs_range[1])
        nodes = np.random.randint(low=nodes_range[0], high=nodes_range[1])
        dropout = np.random.uniform(low=dropout_range[0], high=dropout_range[1])
        batch_size = np.random.randint(low=batch_size_range[0], high=batch_size_range[1])

        print(r, window_size, epochs, features, nodes, dropout)
        
        model = build_part1_RNN(window_size, n_features, nodes, dropout)
        model = training(nikkei225, df_relev_nikkei225, model, window_size, features, epochs, batch_size, False)
        
        mbeta, df_future = best_options(nikkei225, df_relev_nikkei225, model, window_size, features, False)
        
        run = {
            'mean': mbeta,
            'window_size': window_size,
            'epochs': epochs,
            'features': ",".join(features),
            'nodes': nodes,
            'batch_size': batch_size,
            'dropout': dropout
        }
        print(run)
        
        df_mean_beta = df_mean_beta.append(run, ignore_index=True)
    return df_mean_beta
    

if False:
    features = ['Close','ratio5-20', 'ratio20-100', 'Volume']
    # features = ['Close']
    df_mean_beta = init_optimization(nikkei225, df_relev_nikkei225, features)
    # df_mean_beta
    
    
```

    CPU times: user 3 µs, sys: 0 ns, total: 3 µs
    Wall time: 7.15 µs



```python
df_mean_beta.sort_values(by='mean', ascending=False)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-19-9252f7f4f96c> in <module>
    ----> 1 df_mean_beta.sort_values(by='mean', ascending=False)
    

    NameError: name 'df_mean_beta' is not defined

