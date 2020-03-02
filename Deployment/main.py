from flask import Flask, render_template, request, redirect,Markup
import model as mp
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Activation
from keras.layers import LSTM
from sklearn.utils import shuffle
import tensorflow as tf
from keras import losses
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io, base64, os, random
import linear as lm
import lstm as ls

app = Flask(__name__,template_folder='templates')
df = pd.read_csv('Oil and gas for models.csv')
year = df.iloc[:,0].values
df = df.iloc[:, 1:]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/lstm", methods=['GET','POST'])
def lstm():
    if request.method == 'POST':
        a1 = request.form.get("var1")
        a2 = request.form.get("var2")
        a3 = request.form.get("var3")
        a4 = request.form.get("var4")
        a5 = request.form.get("var5")
        a6 = request.form.get("var6")
        a7 = request.form.get("var7")
        a8 = request.form.get("var8")
        print("done")
        parms = [a1,a2,a3,a4,a5,a6,a7,a8]
        df.iloc[83] = parms
        allpred = mp.forecast(df.iloc[:84,:])
        
        print(a1,a2,a3,a4,a5,a6,a7,a8)
        values = list(df.oil_value_2014.values)
        df.iloc[84] = allpred
        model_plot = ''
        plot_url = charta(year[-10:],df.iloc[-10:]['oil_value_2014'].values)
        model_plot = Markup('<img src="data:image/png;base64,{}" width: 360px; height: 288px style="border-radius: 20px ;">'.format(plot_url))

        return render_template('index.html',parm = allpred[0],modelplot=model_plot)
    else:
        return render_template('index.html',parm = "",modelplot="")

def charta(x,valu):
    fig, ax = plt.subplots(figsize=(5,4))
    plt.plot(x,valu,marker='o',markevery=[-1,-2],mec='green',mfc='red',mew=10,ms=3,lw=3,color='purple')
    # let's encode our users
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return(plot_url)

@app.route("/showcase",methods=['GET','POST'])
def showcasingmodels():
    cname = ['Albania', 'Algeria', 'Angola', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belize', 'Benin', 'Bolivia', 'Brazil', 'Brunei', 'Bulgaria', 'Cameroon', 'Canada', 'Chad', 'Chile', 'China', 'Colombia', 'Congo, Dem. Rep.', 'Congo, Rep.', "Cote d'Ivoire", 'Croatia', 'Cuba', 'Czech Republic', 'Czechoslovakia', 'Denmark', 'East Timor', 'Ecuador', 'Egypt, Arab Rep.', 'Equatorial Guinea', 'Estonia', 'Federal Republic of Germany (West Germany)', 'France', 'France including Algeria', 'Gabon', 'Georgia', 'German Democratic Republic (East Germany)', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Hungary', 'India', 'Indonesia', 'Iran, Islamic Rep.', 'Iraq', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyz Republic', 'Libya', 'Lithuania', 'Malaysia', 'Mauritania', 'Mexico', 'Mongolia', 'Morocco', 'Myanmar', 'Netherlands', 'New Zealand', 'Niger', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Pakistan including East Pakistan', 'Papua New Guinea', 'Peru', 'Philippines', 'Poland', 'Qatar', 'Romania', 'Russian Federation', 'Saudi Arabia', 'Senegal', 'Serbia', 'Serbia and Montenegro', 'Slovak Republic', 'Slovenia', 'South Africa', 'South Sudan', 'Soviet Union', 'Spain', 'Sudan', 'Sudan including South Sudan', 'Suriname', 'Sweden', 'Syrian Arab Republic', 'Taiwan', 'Tajikistan', 'Thailand', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uzbekistan', 'Venezuela, RB', 'Vietnam', 'Yemen Arab Rep. (North Yemen)', 'Yemen, Rep.', 'Yugoslavia, Fed. Rep.'] 
    if request.method == 'POST':
        model = request.form.get('model')
        print(model)
        trs = request.form.get('trainr')
        ctyn = request.form.get('city')
        df = pd.read_csv("Oil_and_Gas_1932-2014.csv")
        if model=="LINEAR":
            A,B,trr,trm,trr2,ter,tem,ter2=lm.Accuracy(df,ctyn,float(trs))
            trainplot_ = plottng(A)
            trainplot = Markup('<img style="border-radius: 20px ;margin-top:20px;" src="data:image/png;base64,{}" width: 360px; height: 288px>'.format(trainplot_))
            testplot_ = plottng(B)
            testplot = Markup('<img style="border-radius: 20px ;margin-top:20px;" src="data:image/png;base64,{}" width: 360px; height: 288px>'.format(testplot_))
            mapetr = Markup('<h3 >Model Accuracy Value is : <span style="border-radius: 20px; background-color: azure; padding: 0px 6px;">{}</span></h3>'.format(trr2))
            mapete = Markup('<h3 >Model Accuracy Value is : <span style="border-radius: 20px; background-color: azure; padding: 0px 6px;">{}</span></h3>'.format(ter2))
        
        else:
            # mape_t,rmse_t,mape,rmse
            df1 = pd.read_csv("data1.csv")
            A,B,trr,trm,ter,tem = ls.final(df1,round(float(trs)*82))
            trainplot_ = plotting2(A)
            trainplot = Markup('<img style="border-radius: 20px ;margin-top:20px;" src="data:image/png;base64,{}" width: 360px; height: 288px>'.format(trainplot_))
            testplot_ = plotting2(B)
            testplot = Markup('<img style="border-radius: 20px ;margin-top:20px;" src="data:image/png;base64,{}" width: 360px; height: 288px>'.format(testplot_))
            mapetr = Markup('<h3 >Train MAPE Value is : <span style="border-radius: 20px; background-color: azure; padding: 0px 6px;">{}</span></h3>'.format(trm))
            mapete = Markup('<h3 >Test MAPE Value is : <span style="border-radius: 20px; background-color: azure; padding: 0px 6px;">{}</span></h3>'.format(tem))
        

        return render_template('showcase.html',out = [trr,ter],cnam=cname,trp = trainplot,tsp = testplot,train=mapetr,test=mapete)
    else:
        return render_template('showcase.html',out = ["",""],cnam=cname,trp = "",tsp = "",train="",test="")
    
def plottng(A):
    fig, ax = plt.subplots(figsize=(5,4))
    A.sort_values('Year').plot(x = 'Year',y = ['Actual','Predicted'],ax=ax)
    # plt.plot(A['Year'],A[['Predicted','Year']])
    # let's encode our users
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return(plot_url)

def plotting2(A):
    fig, ax = plt.subplots(figsize=(5,4))
    plt.plot(A['Actual'], label='Actual')
    plt.plot(A['Predicted'], label='Predicted')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return(plot_url)



if __name__ == '__main__':
    app.run()
    