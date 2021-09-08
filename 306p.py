from yahoo_fin.stock_info import *
from PortfolioConstructor import *
import pandas as pd
from datetime import datetime
from datetime import timedelta
from yahoo_earnings_calendar import YahooEarningsCalendar
from numpy import mean, std
import numpy
import statistics
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dateutil.parser
import os
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import pandas_datareader.data as web
import streamlit as st
style.use("ggplot")
DAYS_AHEAD=9
# setting the dates
start_date = datetime.now().date() - timedelta(days=365*2)
end_date = datetime.now().date()
tickers=pd.read_csv("cannabis-stocks.csv")["Symbol"].to_list()[1:]
tickers.append("SPY")
tickers=tickers[::-1]
print(tickers)

ameans=[]
dfcomp=pd.DataFrame()
for ticker in tickers:
    try:
        print(ticker)
        print(get_data(ticker,start_date=start_date,end_date=end_date,interval="1mo")["adjclose"])
        dfcomp[ticker]=get_data(ticker,start_date=start_date,end_date=end_date,interval="1mo")["adjclose"]

    except:
        print(ticker)
        pass
    #print(df)
print(dfcomp)
retscomp = dfcomp.pct_change()
import seaborn as sns
df = sns.load_dataset('iris')

# use the function regplot to make a scatterplot
#sns.regplot(x=retscomp["SPY"], y=retscomp["OXY"])
#plt.title("OXY x SPY Regression Plot")
#plt.show()

# Without regression fit:
#sns.regplot(x=df["sepal_length"], y=df["sepal_width"], fit_reg=False)
#sns.plt.show()

print(dfcomp)
corr = retscomp.corr()
cov=retscomp.cov()
corr.to_csv("cannabis-corr.csv")
cov.to_csv("cannabis-cov.csv")
print(cov)
corr=pd.read_csv("cannabis-corr.csv",index_col=0).dropna(axis='columns')


print(corr)


import numpy as np; np.random.seed(0)
import seaborn as sns
data=corr
import streamlit as st
st.write(data)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
heatmap=ax.pcolor(data,cmap=plt.cm.RdYlGn)
fig.colorbar(heatmap)
ax.set_xticks(np.arange(data.shape[0])+0.5,minor=False)
ax.set_yticks(np.arange(data.shape[1])+0.5,minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()

column_labels=corr.columns
row_labels=corr.index

ax.set_xticklabels(column_labels)
ax.set_yticklabels(row_labels)
plt.xticks(rotation="vertical")
heatmap.set_clim(-1,1)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
spy = get_data("SPY",start_date=start_date,end_date=end_date,interval='1mo')

dfcomp.to_csv("cannabis-comp2.csv")
portfolio_weight_matrix()
input("REFORMAT COLUMNS TO ONLY INCLUDE 2 STOCK WEIGHTS!!! input to continue: ")
matrix_statistics()

'''print(spy)
spy["%change"]=""
ameans=[]
stdvs=[]
divdata=[]
corrs=[]
ticks=[]
returns=[]
sums=[]
betas=[]
for index,y in enumerate(spy["adjclose"]):
        try:
            spy["%change"][index]=(spy["adjclose"][index]-spy["adjclose"][index-1])/spy["adjclose"][index-1]
        except:
            spy["%change"][index]=0
std=(spy["%change"].mean())
print(std)
for stock in tickers:
    ticker=stock


    
    try:
        div=get_dividends(ticker,start_date,end_date)
        print(div)
        divdata.append(True)
    except:
         print("no dividend data")

         divdata.append(False)
    try:
        data=get_data(ticker,start_date,end_date,interval="1mo")

        data["%change"]=""
        data["Dividend"]=""
        divdates=[]

        for index,y in enumerate(data["adjclose"]):


            try:
                data["%change"][index]=(data["adjclose"][index]-data["adjclose"][index-1])/data["adjclose"][index-1]
            except:
                data["%change"][index]=0

        data.to_csv("cannabis-Stocks/{}_price_data.csv".format(ticker))
        div.to_csv("cannabis-Stocks/{}_dividend_data.csv".format(ticker))


        amean=mean(data["%change"])

        std=statistics.stdev(data["%change"].to_list())
        ret=(data["adjclose"][-1]-data["adjclose"][0])/data["adjclose"][0]
        #ret2=(data["adjclose"][-1]-data["adjclose"][-12])/data["adjclose"][-12]
        #cor=data["%change"].corr(spy["%change"])
        try:
            sum=sum(data["%change"].to_list)
            sums.append(sum)
        except:
            sum="Error"

        print(f"{ticker}: Arithmitic Mean:{amean}\tStandard Deviation: {std}\t.S&P Correlation{sum}")
        ameans.append(amean)
        stdvs.append(std)
        #returns.append(ret2)
        corrs.append(ret)
        if stock=="SPY":
            spydev=std
            corr=corr["SPY"]

        for index4,symb in enumerate(corr.index):
            if symb==stock:
                #st.write(corr)
                cor=corr[symb]
                beta=cor*(std/spydev)
        betas.append(beta)

        print(f"{ticker}: Arithmitic Mean:{amean}\tStandard Deviation: {std}\t.S&P Correlation{sum}")
        #print(data,amean)
        ticks.append(ticker)
    except:
        pass
df1 = pd.DataFrame(ticks, columns = ["Ticker"])
df2 = pd.DataFrame(ameans, columns = ["Arithmitic Mean"])
df3= pd.DataFrame(corrs,columns = ["5 Year Return"])
#df6= pd.DataFrame(returns,columns = ["1 Year Return"])
df4 = pd.DataFrame(stdvs, columns = ["Standard Deviation"])
df5 = pd.DataFrame(divdata, columns = ["Dividend Data Availible"])
df7=pd.DataFrame(sums,columns=["Sum"])
df8=pd.DataFrame(betas,columns=["Beta"])



#df4=pd.DataFrame(epsl,columns=["EPS"])
df = df1.join(df2).join(df4).join(df3).join(df5).join(df7).join(df8)
df.to_csv("cannabis-Stocks/306StockStatistics.csv")
import os
os.startfile("cannabis-Stocks/306StockStatistics.csv")
 #   except:
    #    print("no data")"""'''
