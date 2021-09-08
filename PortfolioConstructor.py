import pandas as pd
from numpy import mean, std
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import streamlit as st
import statistics
def portfolio_weight_matrix():
    df=pd.read_csv("cannabis-comp2.csv",index_col=0)

    for index1,stock in enumerate(df):
        str="Change_{}".format(stock)
        df[str]=""
        for index,x in enumerate(df[stock]):

            if index==0:
                df[str][index]=.01
            else:
                df[str][index]=(x-df[stock][index-1])/df[stock][index-1]
    df.to_csv("cannabis-compChange2.csv")
    #os.startfile("cannabis-compChange2.csv")
    tics1=[]
    tickers=df.columns.to_list()
    tickers.append("SPY")
    tickers=tickers[::-1]
    df=tickers
    for tick in df:
        tics1.append(tick)
    tics2=tics1



    df=pd.read_csv("cannabis-compChange2.csv",index_col=0)
    s1wl=[-.2,-.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2]
    s2wl=[]
    for x in s1wl[::-1]:
        s2wl.append(x)
    pairs=[]
    for index1,stock1 in enumerate(tics1):
        for index2,stock2 in enumerate(tics2):
            if stock1!="APA" and stock2!="APA":
                if stock1!=stock2:
                    pairs.append([stock1,stock2])
                    if [stock2,stock1] not in pairs:
                        for index3,sw1 in enumerate(s1wl):
                            sw2=s2wl[index3]
                            str=f"{stock1} {sw1} {stock2} {sw2}"

                            print(str)

                            str1=f"{stock1}"
                            str2=f"{stock2}"
                            if "Change" not in stock1 or "Change" not in stock2:
                                continue

                            index=0
                            #print(df[str1],df[str2])
                            for c1, c2 in zip(df[str1], df[str2]):
                                if index==0:
                                    df[str]=""
                                else:
                                    df[str][index]=(c1*sw1)+(c2*sw2)
                                    #print(df[str][index])
                                index+=1


    df.to_csv("cannabis-portfolioMatrix7.csv")
    #os.startfile("cannabis-portfolioMatrix7.csv")
###MUST REFORMAT MATRIX COLUMNS TO ONLY INCLUDE PORTFOLIO WEIGHT COMBINATIONS OF 2 STOCKS
def matrix_statistics():
    df=pd.read_csv("cannabis-portfolioMatrix7.csv",index_col=0)
    combos=[]
    ameans=[]
    stdvs=[]
    stocks=[]
    for combo in df:
        print(combo)
        mean=(df[combo].mean())
        print(df[combo])
        std=statistics.stdev(df[combo].dropna().to_list())
        print(std)
        combos.append(combo)
        ameans.append(mean)
        stdvs.append(std)
    df1 = pd.DataFrame(combos, columns = ["combo"])
    df2 = pd.DataFrame(ameans, columns = ["Mean"])
    df3= pd.DataFrame(stdvs,columns = ["STD"])
    df=df1.join(df2).join(df3)
    df.to_csv("cannabis-comboResults2.csv")
    df=pd.read_csv("cannabis-comboResults2.csv", index_col=0)
    print(df)
    stocks=[]
    df["Sharpe"]=""
    sharps=[]
    spymean=0.004579
    for index,x in enumerate(df["combo"]):
        df["Sharpe"][index]=(df["Mean"][index]-spymean)/df["STD"][index]
        sharps.append(df["Sharpe"][index])
        if "."in x:
            if [x.split(" ")[0],x.split(" ")[2]] not in stocks:
                try:
                    stocks.append([x.split(" ")[0],x.split(" ")[2]])
                    print([x.split(" ")[0],x.split(" ")[2]])
                except:
                    pass
    print(stocks)
    print(df)
    import plotly.graph_objects as go
    df.to_csv("cannabis-comboResultsSharpe2.csv")
    st.write(df)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["STD"], y=df["Mean"],name="combo",mode='markers',text=df["combo"]))
    st.plotly_chart(fig)
    #os.startfile("cannabis-comboResultsSharpe2.csv")
    fig = go.Figure()
    for pair in stocks:
        stds=[]
        means=[]
        combos=[]
        maxstdindex=0
        maxstd=0
        print(pair)
        for index,x in enumerate(df["combo"]):
            if "." in x or "0" in x:
                if [x.split(" ")[0],x.split(" ")[2]]==pair:

                    combos.append(x)
                    stds.append(df["STD"][index])


                    means.append(df["Mean"][index])
        df1=pd.DataFrame({'STD': stds, 'mean return': means })
        print(df1)
    # plot


        listToStr = ' '.join([str(elem) for elem in pair])
    # Add traces
        fig.add_trace(go.Scatter(x=df1["STD"], y=df1["mean return"],name=listToStr,mode='markers'))

        plt.plot( 'STD', 'mean return', data=df1, linestyle='none', marker='o',label=pair)
        plt.title(pair)
    fig.show()











