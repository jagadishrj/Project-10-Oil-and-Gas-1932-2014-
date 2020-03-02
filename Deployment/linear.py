import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

no_oil=['Afghanistan', 'Armenia', 'Bahamas, The', 'Belgium', 'Bhutan', 'Bosnia and Herzegovina', 'Botswana', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cape Verde', 'Central African Republic', 'Comoros', 'Costa Rica', 'Cyprus', 'Djibouti', 'Dominican Republic', 'El Salvador', 'Eritrea', 'Ethiopia', 'Ethiopia including Eritrea', 'Fiji', 'Finland', 'Gambia, The', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Iceland', 'Ireland', 'Jamaica', 'Kenya', 'Korea, Dem. Rep.', 'Korea, Rep.', 'Lao PDR', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Luxembourg', 'Macedonia, FYR', 'Madagascar', 'Malawi', 'Maldives', 'Mali', 'Malta', 'Mauritius', 'Moldova', 'Montenegro', 'Mozambique', 'Namibia', 'Nepal', 'Nicaragua', 'North Vietnam', 'Panama', 'Paraguay', 'Portugal', 'Rwanda', 'Sierra Leone', 'Singapore', 'Solomon Islands', 'Somalia', 'South Vietnam', 'Sri Lanka', 'Swaziland', 'Switzerland', 'Tanzania', 'Togo', 'Uganda', 'Uruguay', 'Yemen, Dem. Rep. (South Yemen)', 'Zambia', 'Zimbabwe']
Final_oil="oil_value_2014~year+gas_value_nom+oil_price_nom+oil_gas_value_2014+oil_value_nom+gas_prod55_14+gas_value_2014+gas_exports+mult_nom_2014"
def Accuracy(df,cty,tsize):
    df.replace(np.nan,0,inplace=True)
    Final_oil="oil_value_2014~year+gas_value_nom+oil_price_nom+oil_gas_value_2014+oil_value_nom+gas_prod55_14+gas_value_2014+gas_exports+mult_nom_2014"
    col=Final_oil.split('~')[1].split('+');col.append('oil_value_2014')
    data=df.loc[df['cty_name']==cty,col]
    np.random.seed(42)
    Xtrain = data.iloc[0:round(data.shape[0]*tsize),:];Xtest = data.iloc[round(data.shape[0]*tsize):,:];Ytrain=Xtrain.pop('oil_value_2014');Ytest=Xtest.pop('oil_value_2014')
    model=LinearRegression(n_jobs=-1)
    model.fit(Xtrain,Ytrain)
    pred=model.predict(Xtest);predtr=model.predict(Xtrain)
    
    #Creating DataFrame for  train & Test
    train_d=pd.DataFrame();test_d=pd.DataFrame()
    train_d['Actual']=Ytrain;train_d['Predicted']=model.predict(Xtrain);train_d['Year']=Xtrain.year
    test_d['Actual']=Ytest;test_d['Predicted']=model.predict(Xtest);test_d['Year']=Xtest.year

    # Train 
    trainrmse = np.sqrt(np.mean((Ytrain-predtr)**2))
    trainmape=np.mean(abs((Ytrain-predtr))*100/Ytrain)
    trainR2 = model.score(Xtrain,Ytrain)**100
    #Test
    testrmse = np.sqrt(np.mean((Ytest-pred)**2))
    testmape=np.mean(abs((Ytest-pred))*100/Ytest)
    testR2 = model.score(Xtest,Ytest)**100

    return train_d,test_d,trainrmse,trainmape,trainR2,testrmse,testmape,testR2

def lstmf(df,cty,tsize):
    return ""

if __name__ == "__main__":
    df = pd.read_csv("Oil_and_Gas_1932-2014.csv")
    print(Accuracy(df,'India',float('0.3')))
    


