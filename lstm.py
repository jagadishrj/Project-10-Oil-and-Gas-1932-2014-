import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Dropout,Activation
from keras.layers import LSTM, GRU
from sklearn.utils import shuffle
from keras.layers import Flatten    
import tensorflow as tf
from keras import losses
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error ,mean_absolute_error


# data1 = pd.read_csv('data1.csv')


def series_to_supervised(data1, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data1) is list else data1.shape[1]
	df = DataFrame(data1)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def final(data1,n_train):
    values = data1.values
    print(values.shape)
    values = values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled,1,1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)

    values = reframed.values
    # n_train = 60 
    train = values[:n_train]
    test = values[n_train:]
    Xtrain,Ytrain = train[:,:-1],train[:,-1]
    Xtest,Ytest = test[:,:-1],test[:,-1]

    Xtrain=Xtrain.reshape((Xtrain.shape[0],1,Xtrain.shape[1]))
    Xtest=Xtest.reshape((Xtest.shape[0],1,Xtest.shape[1]))


    model = Sequential()

    model.add(LSTM(units = 50 , activation = 'tanh', input_shape=(Xtrain.shape[1],Xtrain.shape[2]), return_sequences = True,kernel_initializer='normal',
                    kernel_regularizer='l2' ))
    model.add(Dropout(0.4))

    #adding a first LSTM layer and some dropout regularisation
    model.add(LSTM(units = 50 , activation = 'tanh', return_sequences = True ))
    model.add(Dropout(0.4))

    #adding a first LSTM layer and some dropout regularisation
    model.add(LSTM(units = 50 , activation = 'tanh', return_sequences = True ))
    model.add(Dropout(0.4))

    #adding a first LSTM layer and some dropout regularisation
    model.add(LSTM(units = 50 , activation = 'tanh', return_sequences = True ))
    model.add(Dropout(0.4))

    #adding a first LSTM layer and some dropout regularisation
    model.add(LSTM(units = 50,activation = 'tanh',))
    model.add(Dropout(0.4))

    #adding output layer
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam' , loss = 'mean_squared_error' , metrics = ['accuracy'])

    #Fitting the Rnn ti the testing set
    model.fit(Xtrain , Ytrain , validation_data = (Xtest, Ytest), epochs = 100 ,batch_size = 20, verbose=0,shuffle =  False)


    # For test

    predicted = model.predict(Xtest)
    XtestRe = Xtest.reshape(Xtest.shape[0],Xtest.shape[2])
    predicted = np.concatenate((predicted,XtestRe[:,1:]),axis=1)
    predicted = scaler.inverse_transform(predicted)

    Ytest = Ytest.reshape(len(Ytest),1)
    Ytest = np.concatenate((Ytest,XtestRe[:,1:]),axis=1)
    Ytest = scaler.inverse_transform(Ytest)

    test_d = pd.concat([pd.Series(predicted[:,0]),pd.Series(Ytest[:,0])],axis=1)
    test_d.columns = ['Predicted','Actual']

    mape = np.mean(np.abs(test_d['Actual']-test_d['Predicted'])/np.abs(test_d['Actual']))  # MAPE imp
    me = np.mean(test_d['Actual']-test_d['Predicted'])             # ME
    mae = np.mean(np.abs(test_d['Actual']-test_d['Predicted']))    
    mpe = np.mean((test_d['Actual']-test_d['Predicted'])/test_d['Actual'])   # MPE
    rmse = np.mean((test_d['Actual']-test_d['Predicted'])**2)**.5  # RMSE imp


    # For Train

    predicted_t = model.predict(Xtrain)
    XtrainRe = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[2])
    predicted_t = np.concatenate((predicted_t,XtrainRe[:,1:]),axis=1)
    predicted_t = scaler.inverse_transform(predicted_t)

    Ytrain = Ytrain.reshape(len(Ytrain),1)
    Ytrain = np.concatenate((Ytrain,XtrainRe[:,1:]),axis=1)
    Ytrain = scaler.inverse_transform(Ytrain)


    train_d = pd.concat([pd.Series(predicted_t[:,0]),pd.Series(Ytrain[:,0])],axis=1)
    train_d.columns = ['Predicted','Actual']

    mape_t = np.mean(np.abs(train_d['Actual']-train_d['Predicted'])/np.abs(train_d['Actual']))  # MAPE imp
    me_t = np.mean(train_d['Actual']-train_d['Predicted'])             # ME
    mae_t = np.mean(np.abs(train_d['Actual']-train_d['Predicted']))    
    mpe_t = np.mean((train_d['Actual']-train_d['Predicted'])/train_d['Actual'])   # MPE
    rmse_t = np.mean((train_d['Actual']-train_d['Predicted'])**2)**.5  # RMSE imp

    return train_d,test_d,rmse_t,mape_t,rmse,mape

# pyplot.plot(train_d['Actual'], label='Actual')
# pyplot.plot(train_d['Predicted'], label='Predicted')
# pyplot.legend()
# pyplot.show()

# pyplot.plot(test_d['Actual'], label='Actual')
# pyplot.plot(test_d['Predicted'], label='Predicted')
# pyplot.legend()
# pyplot.show()

if __name__ == "__main__":
    df = pd.read_csv('data1.csv')
    ntrain = 60
    A,B,_,_,_,_=final(df,ntrain)

    pyplot.plot(A['Actual'], label='Actual')
    pyplot.plot(A['Predicted'], label='Predicted')
    pyplot.legend()
    pyplot.show()

    
