import datetime
import pandas_datareader
import pandas_datareader.data as web
import csv
import time

headers = ['Ticker', 'Available']
f = open('.\Data\log.csv','w')
f_csv = csv.writer(f)
f_csv.writerow(headers)

def Fetch_Data(s):
    start = datetime.datetime(2010, 1, 1) # or start = '1/1/2016'
    end = datetime.date.today()
    try:
        prices = web.DataReader(s, 'yahoo', start, end) # problem
    except pandas_datareader._utils.RemoteDataError:
        row = [s, 'No']
    else:
        row = [s, 'Yes']
        prices.to_csv('.\Data\\' + s + '.csv')
    f = open('.\Data\log.csv', 'a') # log.csv will save data about whether every company's financial data is available
    f_csv = csv.writer(f)
    f_csv.writerow(row)


###  To get names of companies in stock_list.csv
stock_list = list()
with open('.\Pre_Data\stock_list.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        stock_list.append(row[0])
stock_list = stock_list[1:]
###

### To get financial data of every stock in the stock_list.csv
Time = time.time()
length= len(stock_list)
for i in range(length):
    Fetch_Data(stock_list[i])
    print(i, time.time()-Time)
print(time.time()-Time)
###

### Delete empty rows for log.csv
with open('.\Data\log.csv', 'rt')as fin:
    lines = ''
    for line in fin:
        if line != '\n':
            lines += line
with open('.\Data\log.csv', 'wt')as fout:
    fout.write(lines)
#####

### This code is to get all finiancial data of companies in stock_list.csv
# from 2010-1-1 to today (for our data, today is 2020-10-23 since we collected data in that day)
### and this code will also write all data into csv files for all stocks.
### To run this code, you can put stock_list.csv in .\Pre_Data\,
# and write "python FetchData.py" in the terminal. Then you will get 489 stocks' files in .\Data\.
