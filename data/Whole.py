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

def exchange_rate(s):
    try:
        place = s.split(".")[1]
    except IndexError:
        ex_rate = 1
        place='USA'
    else:
        if place == 'TO':
            ex_rate = cad
        elif place == 'L':
            ex_rate = gbp
        elif place == 'PA':
            ex_rate = eur
        elif place == 'HK':
            ex_rate = hkd
        elif place == 'T':
            ex_rate = jpy
    whole_list = [list(), list(),list(),list(),list(),list(),list()]
    with open('.\Data\\' + s + '.csv','r', encoding="utf-8" ) as csvfile:
        reader = csv.reader(csvfile)
        rows0 = [row[0] for row in reader]
    whole_list[0] = rows0[1:]
    for i in range(6):
        with open('.\Data\\' + s + '.csv','r', encoding="utf-8" ) as csvfile:
            reader = csv.reader(csvfile)
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
            # print(len(whole_list[p+1]))
    s_list = [s]*len(whole_list[0])
    whole_list = [s_list] + whole_list
    return whole_list



### To get stocks' name from stock_list_selected.csv, which is come for Yuting Liang who is responsible for
# selecting qualified stocks.
stock_list = list()
currency_list = list()
with open('.\Pre_Data\stock_list_selected.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        stock_list.append(row[0])
        # currency_list.append(row[1])
stock_list = stock_list[1:]
# currency_list = currency_list[1:]
###

Time = time.time()
length= len(stock_list)
whole = [list()]*8
### To use function exchange_rate to exchange order of data and covert all currency to USD in all selected stocks' data.
# Then concatenate all data in to one big list.
for i in range(length):
    whole_list = exchange_rate(stock_list[i])
    for j in range(len(whole)):
        whole[j] = whole[j]+ whole_list[j]


whole= list(map(list, zip(*whole)))
# To use DataFrame to make a table for data of all selected stocks
data = pd.DataFrame(whole,
            columns=['Ticker', 'Date','Open','High','Low','Close','Adj Close','Volume'])
#To make DataFrame data save as csv
data.to_csv('.\whole\whole_selected.csv', index=False)
print(time.time()-Time)

### This code is to put all data of selected stocks together as a whole csv file.
### To run this code, you can put stock_list_selected.csv.csv in .\Pre_Data\ and run FetchData.py firstly..
### Then write "python Whole.py" in the terminal. Finally, you will get whole_selected.csv in .\whole\.