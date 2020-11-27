import datetime
import pandas as pd
import pandas_datareader
import pandas_datareader.data as web
import csv
import time
import os.path

jpy = 104.7600
cad = 1.5548/1.1851
hkd = 7.7498
gbp = 1/1.3076
eur = 1/1.5548
usd = 1

def exchange_rate(s, place):
    if place == 'CAD':
        ex_rate = cad
    elif place == 'GBP':
        ex_rate = gbp
    elif place == 'EUR':
        ex_rate = eur
    elif place == 'HKD':
        ex_rate = hkd
    elif place == 'JPY':
        ex_rate = jpy
    elif place == 'USD':
        ex_rate = usd
    whole_list = [list(), list(),list(),list(),list(),list(),list()]
    with open('.\Data\\' + s + '.csv','r', encoding="utf-8" ) as csvfile:
        reader = csv.reader(csvfile)
        rows0 = [row[0] for row in reader]
    whole_list[0] = rows0[1:]
    for i in range(6):
        with open('.\Data\\' + s + '.csv','r', encoding="utf-8" ) as csvfile:
            reader = csv.reader(csvfile)
            ### To change the order Date, High, Low, Open, Close, Volume, Adj Close to another order
            # 'Date','Open','High','Low','Close','Adj Close','Volume'
            rows = [row[i+1] for row in reader]
            if i ==0:
                p=2
            elif i==1:
                p=3
            elif i==2:
                p=1
            elif i==3:
                p=4
            elif i==4:
                p=6
            elif i==5:
                p=5
            whole_list[p] = [float(row)/ex_rate for row in rows[1:]]
            ### 'Date','Open','High','Low','Close','Adj Close','Volume'
    s_list = [s]*len(whole_list[0])
    whole_list = [s_list] + whole_list
    whole_list = list(map(list, zip(*whole_list)))
    # print(len(whole_list))
    data = pd.DataFrame(whole_list,
                columns=['Ticker', 'Date','Open','High','Low','Close','Adj Close','Volume'])
    data.to_csv('.\list_usd\\' + s + '.csv', index=False)

### To get stocks' name from stock_list.csv
stock_list = list()
currency_list = list()
with open('.\Pre_Data\stock_list.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        stock_list.append(row[0])
        currency_list.append(row[1])
stock_list = stock_list[1:]
currency_list = currency_list[1:]
###

Time = time.time()
length= len(stock_list)
### To use function exchange_rate to exchange order of data and covert all currency to USD in all stocks' data
for i in range(length):
    exchange_rate(stock_list[i], currency_list[i])
print(time.time()-Time)

### This code is to exchange order of data and covert all currency to USD in all stocks' data.
### To run this code, you can put stock_list.csv in .\Pre_Data\ and run FetchData.py firstly.
### Then write "python USD.py" in the terminal. Finally, you will get 489 stocks' files in .\list_usd\.
