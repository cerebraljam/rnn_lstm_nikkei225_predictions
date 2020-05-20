# Using LSTM to predict the Tokyo Stock Exchange variations

One of the first exercise we are proposed to do with Long Short Term Memory models is to try to predict the stock exchange variations.

When we look at the graphs produced by these models, it's quite impressive to see the Training and Test line hug the real price so close, but how good are these predictions?

Hypothesis:
* We only try to predict the next day, because more than that, any model just have no clue where things are going, and the accumulation of unknown just makes it even less reliable. So day traiding scenario only.
* Because values are normalized in order to train the model, the prediction will also be normalized, which means that it will not give a price. We could try to convert the value back to a real value, but is it really necessary? If the objective is just to guess which stock will go up, and how confident we are that it will go up, just a general indication (up/down) could be sufficient.
* Because different areas tend to react differently to the world situation, especially now taht we are during the Covid-19 Pandemic, I decided to concentrate on an area that I hope will be successful: Pharmaceuticals

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

    the current nikkei225.json file is good enough
    Number of symbols in the Nikkei 225 index: 225


# Download the current symbol values from the Yahoo Finance Site


```python
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)
print('today', today.strftime("%Y-%m-%d"), 'tomorrow', tomorrow.strftime("%Y-%m-%d"))

tickers = list(nikkei225.keys()) + ['^N225']
start_date = '2018-01-01'
end_date = tomorrow

all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
```

    today 2020-05-19 tomorrow 2020-05-20



```python
# Downloading the values from Yahoo Finance here
df_raw_nikkei225 = yf.download(tickers, start=start_date, end=end_date)
```

    [*********************100%***********************]  226 of 226 completed



```python
# A bit of cleanup...
df_clean_nikkei225 = df_raw_nikkei225.reindex(all_weekdays)
df_clean_nikkei225 = df_clean_nikkei225.fillna(method='ffill')

df_clean_nikkei225
```




<div>
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
      <th>9503.T</th>
      <th>9531.T</th>
      <th>9532.T</th>
      <th>9602.T</th>
      <th>9613.T</th>
      <th>9735.T</th>
      <th>9766.T</th>
      <th>9983.T</th>
      <th>9984.T</th>
      <th>^N225</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>569.744080</td>
      <td>3270.879639</td>
      <td>1344.925293</td>
      <td>3092.923340</td>
      <td>5212.154297</td>
      <td>1269.765381</td>
      <td>1109.852905</td>
      <td>1591.134033</td>
      <td>2022.888062</td>
      <td>3972.852539</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>569.744080</td>
      <td>3270.879639</td>
      <td>1344.925293</td>
      <td>3092.923340</td>
      <td>5212.154297</td>
      <td>1269.765381</td>
      <td>1109.852905</td>
      <td>1591.134033</td>
      <td>2022.888062</td>
      <td>3972.852539</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>569.744080</td>
      <td>3270.879639</td>
      <td>1344.925293</td>
      <td>3092.923340</td>
      <td>5212.154297</td>
      <td>1269.765381</td>
      <td>1109.852905</td>
      <td>1591.134033</td>
      <td>2022.888062</td>
      <td>3972.852539</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>570.711304</td>
      <td>3285.310059</td>
      <td>1391.219727</td>
      <td>3178.179688</td>
      <td>5323.644043</td>
      <td>1294.899902</td>
      <td>1133.689819</td>
      <td>1622.938477</td>
      <td>2069.541260</td>
      <td>4092.212402</td>
      <td>...</td>
      <td>2487200.0</td>
      <td>1849200.0</td>
      <td>1646900.0</td>
      <td>643400.0</td>
      <td>4249800.0</td>
      <td>800000.0</td>
      <td>797600.0</td>
      <td>997900.0</td>
      <td>17888400.0</td>
      <td>102200.0</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>570.711304</td>
      <td>3285.310059</td>
      <td>1381.674561</td>
      <td>3154.497314</td>
      <td>5370.098145</td>
      <td>1299.554565</td>
      <td>1134.643433</td>
      <td>1646.564697</td>
      <td>2067.675293</td>
      <td>4145.465332</td>
      <td>...</td>
      <td>1917400.0</td>
      <td>1979200.0</td>
      <td>1524300.0</td>
      <td>254900.0</td>
      <td>3359400.0</td>
      <td>685200.0</td>
      <td>494200.0</td>
      <td>731100.0</td>
      <td>11306200.0</td>
      <td>101900.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-05-14</th>
      <td>462.000000</td>
      <td>2305.000000</td>
      <td>663.400024</td>
      <td>3005.000000</td>
      <td>3485.000000</td>
      <td>918.000000</td>
      <td>846.000000</td>
      <td>1157.000000</td>
      <td>1144.000000</td>
      <td>2450.500000</td>
      <td>...</td>
      <td>3607100.0</td>
      <td>817200.0</td>
      <td>987400.0</td>
      <td>329200.0</td>
      <td>4056100.0</td>
      <td>669400.0</td>
      <td>780100.0</td>
      <td>755200.0</td>
      <td>17091200.0</td>
      <td>76900.0</td>
    </tr>
    <tr>
      <th>2020-05-15</th>
      <td>446.000000</td>
      <td>2171.000000</td>
      <td>682.000000</td>
      <td>3020.000000</td>
      <td>3530.000000</td>
      <td>910.000000</td>
      <td>845.000000</td>
      <td>1150.000000</td>
      <td>1185.000000</td>
      <td>2390.000000</td>
      <td>...</td>
      <td>2441600.0</td>
      <td>927000.0</td>
      <td>832200.0</td>
      <td>350000.0</td>
      <td>6569400.0</td>
      <td>512700.0</td>
      <td>1211800.0</td>
      <td>1000200.0</td>
      <td>16681100.0</td>
      <td>75200.0</td>
    </tr>
    <tr>
      <th>2020-05-18</th>
      <td>457.000000</td>
      <td>2288.000000</td>
      <td>709.299988</td>
      <td>3045.000000</td>
      <td>3610.000000</td>
      <td>932.000000</td>
      <td>855.000000</td>
      <td>1187.000000</td>
      <td>1187.000000</td>
      <td>2416.500000</td>
      <td>...</td>
      <td>3044800.0</td>
      <td>976700.0</td>
      <td>869500.0</td>
      <td>352300.0</td>
      <td>3188600.0</td>
      <td>325600.0</td>
      <td>671600.0</td>
      <td>592800.0</td>
      <td>25380600.0</td>
      <td>71900.0</td>
    </tr>
    <tr>
      <th>2020-05-19</th>
      <td>463.000000</td>
      <td>2246.000000</td>
      <td>732.900024</td>
      <td>3060.000000</td>
      <td>3675.000000</td>
      <td>953.000000</td>
      <td>869.000000</td>
      <td>1214.000000</td>
      <td>1207.000000</td>
      <td>2515.000000</td>
      <td>...</td>
      <td>3700200.0</td>
      <td>900400.0</td>
      <td>894200.0</td>
      <td>433800.0</td>
      <td>5696500.0</td>
      <td>577600.0</td>
      <td>926700.0</td>
      <td>873300.0</td>
      <td>41866700.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-05-20</th>
      <td>463.000000</td>
      <td>2246.000000</td>
      <td>732.900024</td>
      <td>3060.000000</td>
      <td>3675.000000</td>
      <td>953.000000</td>
      <td>869.000000</td>
      <td>1214.000000</td>
      <td>1207.000000</td>
      <td>2515.000000</td>
      <td>...</td>
      <td>3700200.0</td>
      <td>900400.0</td>
      <td>894200.0</td>
      <td>433800.0</td>
      <td>5696500.0</td>
      <td>577600.0</td>
      <td>926700.0</td>
      <td>873300.0</td>
      <td>41866700.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>623 rows × 1356 columns</p>
</div>




```python
# keeping a dataframe with just the close values
df_close_nikkei225 = df_clean_nikkei225['Close']

# here I am adding few values, mainly rolling averages over 5,20 and 100 days, 
# then the ratio between the 5 and 20 days rolling average, then 20 and 100 days
# I am also adding value of the nikkei index to each entry.
for symbol in [x for x in nikkei225.keys()]:
    df_clean_nikkei225[('RollAvg5', symbol)] = df_close_nikkei225.loc[:, symbol].rolling(window=5).mean()
    df_clean_nikkei225[('RollAvg20', symbol)] = df_close_nikkei225.loc[:, symbol].rolling(window=20).mean()
    df_clean_nikkei225[('RollAvg100', symbol)] = df_close_nikkei225.loc[:, symbol].rolling(window=100).mean()
    df_clean_nikkei225[('N225', symbol)] = df_close_nikkei225.loc[:, '^N225']
    
    df_clean_nikkei225[('ratio5-20', symbol)] = df_clean_nikkei225[('RollAvg5', symbol)] / df_clean_nikkei225[('RollAvg20', symbol)]
    df_clean_nikkei225[('ratio20-100', symbol)] = df_clean_nikkei225[('RollAvg20', symbol)] / df_clean_nikkei225[('RollAvg100', symbol)]
    
    
# keeping a dataframe with the relevant values used for our model
df_relev_nikkei225 = df_clean_nikkei225[['Close', 'High', 'Low', 'Volume', 'RollAvg5', 'RollAvg20', 'RollAvg100', 'ratio5-20', 'ratio20-100', 'N225']]
df_relev_nikkei225.tail(-100)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Close</th>
      <th>...</th>
      <th colspan="10" halign="left">N225</th>
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
      <th>2018-05-21</th>
      <td>570.0</td>
      <td>4125.0</td>
      <td>1378.500000</td>
      <td>2921.0</td>
      <td>6130.0</td>
      <td>1161.0</td>
      <td>1109.0</td>
      <td>1660.0</td>
      <td>1822.0</td>
      <td>4205.0</td>
      <td>...</td>
      <td>23002.369141</td>
      <td>23002.369141</td>
      <td>23002.369141</td>
      <td>23002.369141</td>
      <td>23002.369141</td>
      <td>23002.369141</td>
      <td>23002.369141</td>
      <td>23002.369141</td>
      <td>23002.369141</td>
      <td>23002.369141</td>
    </tr>
    <tr>
      <th>2018-05-22</th>
      <td>571.0</td>
      <td>4065.0</td>
      <td>1364.000000</td>
      <td>2904.0</td>
      <td>6140.0</td>
      <td>1154.0</td>
      <td>1106.0</td>
      <td>1661.0</td>
      <td>1834.0</td>
      <td>4184.0</td>
      <td>...</td>
      <td>22960.339844</td>
      <td>22960.339844</td>
      <td>22960.339844</td>
      <td>22960.339844</td>
      <td>22960.339844</td>
      <td>22960.339844</td>
      <td>22960.339844</td>
      <td>22960.339844</td>
      <td>22960.339844</td>
      <td>22960.339844</td>
    </tr>
    <tr>
      <th>2018-05-23</th>
      <td>562.0</td>
      <td>4060.0</td>
      <td>1302.500000</td>
      <td>2891.0</td>
      <td>6050.0</td>
      <td>1142.0</td>
      <td>1099.0</td>
      <td>1642.0</td>
      <td>1842.0</td>
      <td>4109.0</td>
      <td>...</td>
      <td>22689.740234</td>
      <td>22689.740234</td>
      <td>22689.740234</td>
      <td>22689.740234</td>
      <td>22689.740234</td>
      <td>22689.740234</td>
      <td>22689.740234</td>
      <td>22689.740234</td>
      <td>22689.740234</td>
      <td>22689.740234</td>
    </tr>
    <tr>
      <th>2018-05-24</th>
      <td>551.0</td>
      <td>4110.0</td>
      <td>1284.500000</td>
      <td>2915.0</td>
      <td>6060.0</td>
      <td>1119.0</td>
      <td>1087.0</td>
      <td>1625.0</td>
      <td>1818.0</td>
      <td>4034.0</td>
      <td>...</td>
      <td>22437.009766</td>
      <td>22437.009766</td>
      <td>22437.009766</td>
      <td>22437.009766</td>
      <td>22437.009766</td>
      <td>22437.009766</td>
      <td>22437.009766</td>
      <td>22437.009766</td>
      <td>22437.009766</td>
      <td>22437.009766</td>
    </tr>
    <tr>
      <th>2018-05-25</th>
      <td>547.0</td>
      <td>4090.0</td>
      <td>1248.500000</td>
      <td>2929.0</td>
      <td>6040.0</td>
      <td>1136.0</td>
      <td>1085.0</td>
      <td>1649.0</td>
      <td>1828.0</td>
      <td>4029.0</td>
      <td>...</td>
      <td>22450.789062</td>
      <td>22450.789062</td>
      <td>22450.789062</td>
      <td>22450.789062</td>
      <td>22450.789062</td>
      <td>22450.789062</td>
      <td>22450.789062</td>
      <td>22450.789062</td>
      <td>22450.789062</td>
      <td>22450.789062</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <td>19914.779297</td>
      <td>19914.779297</td>
      <td>19914.779297</td>
      <td>19914.779297</td>
      <td>19914.779297</td>
      <td>19914.779297</td>
      <td>19914.779297</td>
      <td>19914.779297</td>
      <td>19914.779297</td>
      <td>19914.779297</td>
    </tr>
    <tr>
      <th>2020-05-15</th>
      <td>446.0</td>
      <td>2171.0</td>
      <td>682.000000</td>
      <td>3020.0</td>
      <td>3530.0</td>
      <td>910.0</td>
      <td>845.0</td>
      <td>1150.0</td>
      <td>1185.0</td>
      <td>2390.0</td>
      <td>...</td>
      <td>20037.470703</td>
      <td>20037.470703</td>
      <td>20037.470703</td>
      <td>20037.470703</td>
      <td>20037.470703</td>
      <td>20037.470703</td>
      <td>20037.470703</td>
      <td>20037.470703</td>
      <td>20037.470703</td>
      <td>20037.470703</td>
    </tr>
    <tr>
      <th>2020-05-18</th>
      <td>457.0</td>
      <td>2288.0</td>
      <td>709.299988</td>
      <td>3045.0</td>
      <td>3610.0</td>
      <td>932.0</td>
      <td>855.0</td>
      <td>1187.0</td>
      <td>1187.0</td>
      <td>2416.5</td>
      <td>...</td>
      <td>20133.730469</td>
      <td>20133.730469</td>
      <td>20133.730469</td>
      <td>20133.730469</td>
      <td>20133.730469</td>
      <td>20133.730469</td>
      <td>20133.730469</td>
      <td>20133.730469</td>
      <td>20133.730469</td>
      <td>20133.730469</td>
    </tr>
    <tr>
      <th>2020-05-19</th>
      <td>463.0</td>
      <td>2246.0</td>
      <td>732.900024</td>
      <td>3060.0</td>
      <td>3675.0</td>
      <td>953.0</td>
      <td>869.0</td>
      <td>1214.0</td>
      <td>1207.0</td>
      <td>2515.0</td>
      <td>...</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
    </tr>
    <tr>
      <th>2020-05-20</th>
      <td>463.0</td>
      <td>2246.0</td>
      <td>732.900024</td>
      <td>3060.0</td>
      <td>3675.0</td>
      <td>953.0</td>
      <td>869.0</td>
      <td>1214.0</td>
      <td>1207.0</td>
      <td>2515.0</td>
      <td>...</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
      <td>20433.449219</td>
    </tr>
  </tbody>
</table>
<p>523 rows × 2254 columns</p>
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
def window_transform_3D_series(df_X, df_y, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for ii in range(len(df_X) - (window_size)):
        X.append(list(df_X.iloc[ii:ii+window_size].values))
        y.append(df_y.iloc[ii+window_size])

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
    objective='Close'
    n_features = len(features)

    for symbol in list(df_close_nikkei225.keys())[:1]:
        print(symbol)
        df_X = df_relev_nikkei225.loc[:, [(x, symbol) for x in features]]
        df_y = df_relev_nikkei225.loc[:, (objective, symbol)]

        df_X_scaled=((df_X-df_X.mean())/df_X.std())
        df_y_scaled=((df_y-df_y.mean())/df_y.std())

        X,y = window_transform_3D_series(df_X_scaled.fillna(0), df_y_scaled.fillna(0), window_size)
```

## Initializing the model


```python
def training(nikkei225, df_relev_nikkei225, areas, model, window_size, features, objective, epochs, batch_size, verbose=True):
    n_features = len(features)
       
    if len(areas) == 0:
        array = [x for x in nikkei225.keys()]
    else:
        array = [x for x in nikkei225.keys() if nikkei225[x]['area'] in areas]
        
    np.random.shuffle(array)
    
    for s in range(len(array)):
        symbol = array[s]
        df_X = df_relev_nikkei225.loc[:, [(x, symbol) for x in features]]
        df_y = df_relev_nikkei225.loc[:, (objective, symbol)]

        df_X_scaled=(df_X-df_X.mean())/df_X.std()
        df_y_scaled=(df_y-df_y.mean())/df_y.std()

        X,y = window_transform_3D_series(df_X_scaled.fillna(0), df_y_scaled.fillna(0), window_size)

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
        plt.plot(df_X_scaled[objective].reset_index(drop=True),color = 'k')


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
window_size = 4
features = [
    'Close',
    'Low',
    'High',
    #'RollAvg5', 
    #'RollAvg100',
    'ratio5-20',
    #'ratio20-100',
    'Volume',
    'N225'
]
n_features = len(features)

objective='Close'
areas = ['Pharmaceuticals']#, 'Services']

error_objective = 0.02

epochs = 50
batch_size = 50
nodes = 5
dropout = 0.2

model = build_part1_RNN(window_size, n_features, nodes, dropout)

model = training(nikkei225, df_relev_nikkei225, areas, model, window_size, features, objective, epochs, batch_size, True)
```

    * 4503.T ASTELLAS PHARMA INC. training error = 0.051 (False), testing error: 0.115 (False)
    * 4151.T KYOWA KIRIN CO., LTD. training error = 0.041 (False), testing error: 0.744 (False)
    * 4502.T TAKEDA PHARMACEUTICAL CO., LTD. training error = 0.019 (True), testing error: 0.148 (False)
    * 4519.T CHUGAI PHARMACEUTICAL CO., LTD. training error = 0.014 (True), testing error: 0.421 (False)
    * 4506.T SUMITOMO DAINIPPON PHARMA CO., LTD. training error = 0.032 (False), testing error: 0.090 (False)
    * 4523.T EISAI CO., LTD. training error = 0.051 (False), testing error: 0.028 (False)
    * 4578.T OTSUKA HOLDINGS CO., LTD. training error = 0.056 (False), testing error: 0.068 (False)
    * 4507.T SHIONOGI & CO., LTD. training error = 0.054 (False), testing error: 0.210 (False)
    * 4568.T DAIICHI SANKYO CO., LTD. training error = 0.033 (False), testing error: 0.104 (False)



![png](tse_lstm_predictions_files/tse_lstm_predictions_19_1.png)



![png](tse_lstm_predictions_files/tse_lstm_predictions_19_2.png)


Impressive! It kind of follow the real values, like for all the LSTM examples! ^_^

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
def strategy_predict_higher(predicted, previous_prediction):
    decision = False
    difference = predicted - previous_prediction
            
    if difference <= 0:
        guess = 'Down'.format(predicted)
    else:
        guess = 'Up'.format(predicted)
        
        if difference > predicted:
            decision = True
                    
    return decision, guess

def strategy_random(predicted, previous_prediction):   
    difference = predicted - previous_prediction
    
    if difference < 0:
        guess = 'Down'.format(predicted)
    else:
        guess = 'Up'.format(predicted)

    decision = random.randint(0,1)
    
    return decision, guess
```


```python
def best_options(nikkei225, df_relev_nikkei225, areas, model, window_size, features, objective, strategy, verbose=False):
    
    df_future = pd.DataFrame(columns = ['date', 'symbol', 'name', 'area', 'close', 'close_normalized', 'prediction', 'direction', 'buy', 'a', 'b', 'mean_beta', 'capital'])
    n_features = len(features)
    aa = 1
    bb = 1
    
    if len(areas) == 0:
        array = [x for x in nikkei225.keys()]
    else:
        array = [x for x in nikkei225.keys() if nikkei225[x]['area'] in areas]
        
    np.random.shuffle(array)
    
    for s in range(len(array)):
        symbol = array[s]
        
        df_X = df_relev_nikkei225.loc[:, [(x, symbol) for x in features]]
        df_y = df_relev_nikkei225.loc[:, (objective, symbol)]

        df_X_scaled=(df_X-df_X.mean())/df_X.std()
        df_y_scaled=(df_y-df_y.mean())/df_y.std()

        X,y = window_transform_3D_series(df_X_scaled.fillna(0), df_y_scaled.fillna(0), window_size)

        train_test_split = int(np.ceil(4*len(y)/float(5)))   # set the split point (80%)

        # keep the last chunk for testing
        X_test = X[train_test_split:,:]
        y_test = y[train_test_split:]

        # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, n_features] 
        X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, n_features)))

        # generate predictions for training
        test_predict = model.predict(X_test)

        df_X_scaled_reindexed = df_X_scaled.reset_index()
        df_y_scaled_reindexed = df_y_scaled.reset_index()
        
        df_y_reindexed = df_y.reset_index()

        a = 1
        b = 1
        capital = 0
    
        for t in range(1, len(y_test-1)):
            
            scaled_sr = df_y_scaled_reindexed.iloc[train_test_split + t + window_size-1][['index', objective]]
            raw_sr = df_y_reindexed.iloc[train_test_split + t + window_size-1][['index', objective]]
            raw_tomorrow_sr = df_y_reindexed.iloc[train_test_split + t + window_size][['index', objective]]
            
            rday = str(raw_sr['index'].values[0]).split('T')[0]
            rtomorrow = str(raw_tomorrow_sr['index'].values[0]).split('T')[0]
            
            scaled_close = scaled_sr[(objective, symbol)]
            raw_close = raw_sr[(objective, symbol)]

            predicted = test_predict[t][0]
            previous_prediction = test_predict[t-1][0]
            
            
            if strategy == 'higher':
                decision, guess = strategy_predict_higher(predicted, previous_prediction)
            elif strategy == 'random':
                decision, guess = strategy_random(predicted, previous_prediction)
            else: # if the strategy is undefined, we don't buy anything
                decision = False
                guess = 'Undefined'
            
            
            if len(y_test) > t+1:
                scaled_next = y_test[t]
                raw_next = raw_tomorrow_sr[(objective, symbol)]

                profit =  raw_next - raw_close

                if decision: # We decide to buy
                    
                    capital += profit # Adjust how much capital was gained/lost
                    
                    if profit > 0: # Keep track of how many time we were successful...
                        a+=1 # local for the symbol
                        aa+=1 # global for all symbols
                    if profit <= 0: # or not successful
                        b+=1
                        bb+=1

                    if False:
                        if profit > 0:
                            print(scaled_close, raw_close, predicted, guess, decision, profit)
                            
                else: # When we decide not to buy...
                    if profit < 0: # and we were right...
                        a+=1
                        aa+=1
                    if profit > 0 and False: # lost opportunity but I think we shouldn't be punished for it
                        b+=1
                        bb+=1
                    
                    
            else: # Exit condition: We can't confirm the prediction with the value of the next day
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
                    print("{} {} {} (buy: {}) mean_beta: {:0.3f}, capital: {}".format(item['symbol'], item['name'], item['area'], item['buy'], item['mean_beta'], capital))


    return mean_beta(aa,bb), df_future
```


```python
strategy = 'higher'

mbeta, df_future = best_options(nikkei225, df_relev_nikkei225, areas, model, window_size, features, objective, strategy, True)
print('\nmean_beta: {:0.4f}'.format(mbeta))
```

    4507.T SHIONOGI & CO., LTD. Pharmaceuticals (buy: True) mean_beta: 0.795, capital: -62.0
    4506.T SUMITOMO DAINIPPON PHARMA CO., LTD. Pharmaceuticals (buy: True) mean_beta: 0.637, capital: -514.0
    4502.T TAKEDA PHARMACEUTICAL CO., LTD. Pharmaceuticals (buy: True) mean_beta: 0.685, capital: 457.0
    4578.T OTSUKA HOLDINGS CO., LTD. Pharmaceuticals (buy: True) mean_beta: 0.681, capital: -325.0
    
    mean_beta: 0.8088


That last mean_beta value is an important indicator. It means that the model and conditions was successful ~80.8% of the time with the current model and parameters.

It is important to take into consideration how "successes" and "failures" is defined:
* if an increase is predicted, we decide to buy, and there is a profit (>0) observed the next day: `success`
* if an increase is predicted, we decide to buy, and there is a loss (<=0) observed the next day: `failure`
* if a loss is predicted, we decide not to buy, and there is a loss (<=0) observed the next day: `success`
* if a loss is predicted, we decide not to buy, and there is a profit (>0) the next day: `neither success/failure`. just a lost opportunity.

This last situation does have an impact on the success rate.

## According to our model, which stock is expected to rise tomorrow?


```python
budget_limit = 3000 # maximum price willing to pay per share

df_future[(df_future['buy'] == True) & (df_future['close'] < budget_limit)].sort_values(['mean_beta'], ascending=False)
```




<div>
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
      <td>2020-05-19</td>
      <td>4506.T</td>
      <td>SUMITOMO DAINIPPON PHARMA CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>1401.0</td>
      <td>-1.364908</td>
      <td>-1.054401</td>
      <td>Up</td>
      <td>True</td>
      <td>58</td>
      <td>33</td>
      <td>0.6373626373626373</td>
      <td>-514.0</td>
    </tr>
  </tbody>
</table>
</div>



In this case, the model is telling us that we should not buy any share because it believes that one will be profitable enough.


```python
# this cell graphes all the shares identified as potentials by the model.
for i, r in df_future[(df_future['buy'] == True) & (df_future['close'] < budget_limit)].sort_values(['mean_beta'], ascending=False).iterrows():
    graph_symbol(r['symbol'], df_close_nikkei225)
```

    4506.T



![png](tse_lstm_predictions_files/tse_lstm_predictions_29_1.png)


## How about the other stocks, what should happen to them?

Ok, the model is telling us that we should not buy anything, but does this mean that no share will be going up? Here, because I did set a maximum budget per share to 2000¥, anything that is more expensive will not be displayed.

The following cell is displaying all the shares that are predicted to go up, regardless of the budget, or the expectations of profits.


```python
df_future[df_future['direction'] == 'Up'].sort_values(['mean_beta'], ascending=False).head(20)
```




<div>
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
      <th>6</th>
      <td>2020-05-19</td>
      <td>4568.T</td>
      <td>DAIICHI SANKYO CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>8519.0</td>
      <td>2.184459</td>
      <td>1.629790</td>
      <td>Up</td>
      <td>False</td>
      <td>55</td>
      <td>1</td>
      <td>0.9821428571428571</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-05-19</td>
      <td>4151.T</td>
      <td>KYOWA KIRIN CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>2621.0</td>
      <td>1.926043</td>
      <td>1.495396</td>
      <td>Up</td>
      <td>False</td>
      <td>55</td>
      <td>3</td>
      <td>0.9482758620689655</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2020-05-19</td>
      <td>4507.T</td>
      <td>SHIONOGI &amp; CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>5743.0</td>
      <td>-0.783053</td>
      <td>-0.591052</td>
      <td>Up</td>
      <td>True</td>
      <td>62</td>
      <td>16</td>
      <td>0.7948717948717948</td>
      <td>-62.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-05-19</td>
      <td>4502.T</td>
      <td>TAKEDA PHARMACEUTICAL CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>4085.0</td>
      <td>-0.438334</td>
      <td>-0.374585</td>
      <td>Up</td>
      <td>True</td>
      <td>61</td>
      <td>28</td>
      <td>0.6853932584269663</td>
      <td>457.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-05-19</td>
      <td>4578.T</td>
      <td>OTSUKA HOLDINGS CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>4329.0</td>
      <td>-0.650294</td>
      <td>-0.474505</td>
      <td>Up</td>
      <td>True</td>
      <td>47</td>
      <td>22</td>
      <td>0.6811594202898551</td>
      <td>-325.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-05-19</td>
      <td>4506.T</td>
      <td>SUMITOMO DAINIPPON PHARMA CO., LTD.</td>
      <td>Pharmaceuticals</td>
      <td>1401.0</td>
      <td>-1.364908</td>
      <td>-1.054401</td>
      <td>Up</td>
      <td>True</td>
      <td>58</td>
      <td>33</td>
      <td>0.6373626373626373</td>
      <td>-514.0</td>
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

## The trading day is over. How much money did we make?

So the night before the model gave us a bunch of shares to select from. Let's say that I bought 100 of each. Now that the day of training is over (passed 15:00 JST), how much money did I make?


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

    yesterday 2020-05-19 now 2020-05-20 tomorrow 2020-05-21
    [*********************100%***********************]  6 of 6 completed





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>4151.T</th>
      <th>4502.T</th>
      <th>4506.T</th>
      <th>4507.T</th>
      <th>4568.T</th>
      <th>4578.T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-19</th>
      <td>2621.0</td>
      <td>4085.0</td>
      <td>1401.0</td>
      <td>5743.0</td>
      <td>8519.0</td>
      <td>4329.0</td>
    </tr>
    <tr>
      <th>2020-05-20</th>
      <td>2664.0</td>
      <td>4125.0</td>
      <td>1407.0</td>
      <td>5803.0</td>
      <td>8571.0</td>
      <td>4353.0</td>
    </tr>
    <tr>
      <th>2020-05-21</th>
      <td>2664.0</td>
      <td>4125.0</td>
      <td>1407.0</td>
      <td>5803.0</td>
      <td>8571.0</td>
      <td>4353.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
series_capital = (df_clean_focus_nikkei2225['Close'].loc[tomorrow] - df_clean_focus_nikkei2225['Close'].loc[today]) * 100
print('Symbols with profits: {}. with losses: {}. Capital Gain/loss: {:0.0f}¥'.format(series_capital.where(series_capital>0).count(), series_capital.where(series_capital<=0).count(), series_capital.sum()))
```

    Symbols with profits: 6. with losses: 0. Capital Gain/loss: 22500¥


In the 6 symbols that were predicted to go up, 6 did... and 0 didn't.
If we had bought 100 shares of each (minimum that we can buy on the TSE), we would end the day with a profit of 22500¥... Not quite enough to retire yet, but still a gain.

### How performant is this prediction? 

Actually, I am quite surprised by this outcome. It must have been an exceptional day on the stock market, because success/fail tend to be close to the average. A success rate of 100% is really unlikely.

### How would a random strategy compete against this heavy model?


```python
strategy = 'random'

mbeta, df_future = best_options(nikkei225, df_relev_nikkei225, areas, model, window_size, features, objective, strategy, True)
print('mean_beta: ', mbeta)
```

    4523.T EISAI CO., LTD. Pharmaceuticals (buy: 1) mean_beta: 0.653, capital: 799.0
    4519.T CHUGAI PHARMACEUTICAL CO., LTD. Pharmaceuticals (buy: 1) mean_beta: 0.578, capital: 1350.0
    4578.T OTSUKA HOLDINGS CO., LTD. Pharmaceuticals (buy: 1) mean_beta: 0.635, capital: 802.0
    4507.T SHIONOGI & CO., LTD. Pharmaceuticals (buy: 1) mean_beta: 0.635, capital: 9.0
    4568.T DAIICHI SANKYO CO., LTD. Pharmaceuticals (buy: 1) mean_beta: 0.602, capital: 1280.0
    mean_beta:  0.6184049079754601


Following the same conditions, a random selection would give us a success rate of 62%. A Random selection seems quite be quite comparable to our heavy LSTM model!

### How about if we don't buy anything?


```python
strategy = 'buy nothing!'

mbeta, df_future = best_options(nikkei225, df_relev_nikkei225, areas, model, window_size, features, objective, strategy, True)
print('mean_beta: ', mbeta)
```

    mean_beta:  0.9980119284294234


We can't be wrong if we don't lose money... since we consider a missed opportunity just as good as not losing our investment.

# Conclusion

The predictions from the LSTM RNN seems to be following the reality, but if we try to use these predictions to take purchasing decisions, the results should be close to a coin toss.

What is scary is that a random selection manage to get a 62% success rate.

Can we use this in real life? I didn't. Not sure I will, especially not blindly.

# Optimization section

This section is mainly to try multiple parameters over and over, attempting to find the best parameters for the model.

It is currently set not to run.


```python
%%time 

def init_optimization(nikkei225, df_relev_nikkei225, areas, features, objective, strategy):
    df_mean_beta = pd.DataFrame(columns=['mean', 'window_size', 'epochs', 'features', 'nodes', 'dropout'])
    
    window_size_range = [3, 6]
    epochs_range = [30, 60]
    nodes_range = [5, 30]
    batch_size_range = [40, 75]
    dropout_range = [0.0, 0.2]
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
        model = training(nikkei225, df_relev_nikkei225, areas, model, window_size, features, objective, epochs, batch_size, False)
        
        mbeta, df_future = best_options(nikkei225, df_relev_nikkei225, areas, model, window_size, features, objective, strategy, False)
        
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
    

if True:
    areas = ['Pharmaceuticals']#, 'Foods', 'Services']
    
    features = [
        'Close',
        'Low',
        'High',
        #'RollAvg5', 
        #'RollAvg100',
        'ratio5-20',
        'ratio20-100',
        'Volume',
        'N225'
    ]
    
    objective = 'Close'
    strategy = 'higher'
    df_mean_beta = init_optimization(nikkei225, df_relev_nikkei225, areas, features, objective, strategy)
    
    
```


```python
df_mean_beta.sort_values(by='mean', ascending=False)
```

```
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
```
