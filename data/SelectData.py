import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from glob import glob
import os.path

def MakePlot(strfolder,ticker,x,y):
    plt.plot(x,y,linewidth=0.75)
    plt.title(ticker)
    plt.savefig(strfolder+ticker+'.png',bbox_inches='tight',pad_inches=0.05)
    plt.close()

def ExtractClosingPrice(strfilename,colname):
    table = np.genfromtxt(strfilename,dtype=None,delimiter=',',names=True)#,missing_values='',filling_values='')
    date_col = table['Date']
    close_col = table[colname]
    x = []
    n = len(date_col)
    for i in range(n):
        dt = datetime.strptime(date_col[i].decode(),'%Y-%m-%d')
        x.append(dt)
    return x, close_col, n

def PlotAllTimeSeries(strfolderin,strfolderout,strlogfile,colname='Close'):
    filelist = [os.path.basename(filename) for filename in glob(strfolderin+'*.csv')]
    fOut = open(strlogfile,'w')
    fOut.write('Ticker,Count,Start,End\n')
    for filename in filelist:
        x, y, n = ExtractClosingPrice(strfolderin+filename,colname)
        ticker = filename.replace('.csv','')
        MakePlot(strfolderout,ticker,x,y)
        start = min(x)
        end = max(x)
        fOut.write(','.join([ticker,str(n),str(start),str(end)])+'\n')
    fOut.close()

strfolderin = '/Data/Processed/'
strfolderout = 'Plots/'
PlotAllTimeSeries(strfolderin,strfolderout,strfolderout+'log.csv')


