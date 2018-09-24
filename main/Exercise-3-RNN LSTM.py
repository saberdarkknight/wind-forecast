# -*- coding: utf-8 -*-
"""
Created on Sun May 13 00:02:06 2018

@author: Ghali Yakoub
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the training dataset
dataset_train= pd.read_csv('TrainData.csv')
training_set = dataset_train.iloc[:,1:2 ].values

#Creating the 48hours timesteps structure
X_train=[]
y_train=[]
for i in range(48 , 16080):
    X_train.append(training_set[i-48:i, 0])
    y_train.append(training_set[i, 0])
X_train, y_train =np.array(X_train),np.array (y_train)
#Reshaping
X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1))

# Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# Intilizing the RNN
regressor= Sequential()
regressor.add(LSTM(units=40, return_sequences=True, input_shape= ( X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.15))
regressor.add(LSTM(units=40, return_sequences=True))
regressor.add(Dropout(rate=0.15))
regressor.add(LSTM(units=40, return_sequences=True))
regressor.add(Dropout(rate=0.15))
regressor.add(LSTM(units=40))
regressor.add(Dropout(rate=0.15))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs= 10, batch_size=32 )


#prediction
dataset_test= pd.read_csv('Solution.csv')
test_set = dataset_test.iloc[:,1:2 ].values
y_test=test_set
dataset_total=pd.concat((dataset_train['POWER'],dataset_test['POWER']), axis= 0, )
inputs= dataset_total[len(dataset_total)-len(dataset_test)-48:].values
inputs=inputs.reshape(-1,1)

X_test=[]
for i in range(48 , 768):
    X_test.append(inputs[i-48:i, 0])
X_test =np.array(X_test)
X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))
y_pred_test = regressor.predict(X_test)
y_pred_train= regressor.predict(X_train)

import math 
#Root mean square error
error_test = y_test-y_pred_test
error_test_sq=np.square (error_test)
RMSE_test=math.sqrt(sum(error_test_sq)/error_test.shape[0])
 

error_train = y_train-y_pred_train
error_train_sq=np.square (error_train)
RMSE_train=math.sqrt( sum(sum(error_train_sq)/error_train.shape[0]) /error_train.shape[0])

Overfitting_indicator_test =RMSE_train/RMSE_test

result=([['RMSE_Train','RMSE_Test_set','Overfitting_test'],
         [ RMSE_train, RMSE_test ,Overfitting_indicator_test]])
result=np.array(result)


#Efficiency index
#Yav_test=(1/error_test.shape[0])*sum(y_test)
#Yav_test=float(Yav_test)
#ST_test=sum((y_test-Yav_test)**2)
##ST_test=float(ST_test)
#EI_test= (ST_test-sum(error_test_sq))/ST_test
#EI_test=float(EI_test)



#Yav_train=(1/error_train.shape[0])*sum(y_train)
#Yav_train=float(Yav_train)
#ST_train=sum((y_train-Yav_train)**2)
#ST_train=float(ST_train)
#EI_train= (ST_train-sum(error_train_sq))/ST_train
#EI_train=float(EI_train)
#
#Performance_parameters=([['.','Train_set','Test_set'],
#                         ['RMSE', RMSE_train, RMSE_test],
#                         ['Mean',Yav_train,Yav_test],
#                        ['EI',EI_train,EI_test]])
#Performance_parameters=np.array(Performance_parameters)

import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

# Visualising the Regression results
params = {'legend.fontsize': 40}
plt.rcParams.update(params)
x=np.arange(720)
fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(x,y_test[:],c='b',label='observed',fillstyle='none',  linewidth=6.0)
ax.plot(x,y_pred_test[:],c='r',label='predicted',  linewidth=6.0)
ax.xaxis.set_ticks([i*60 for i in range(0,13)])
ax.grid()
ax.set_ylabel('Power Generation', fontsize=44)
ax.set_xlabel('Time', fontsize=44)
ax.tick_params(labelsize=44)
plt.legend(loc=2)
plt.title('RNN', fontsize=30)
plt.draw()



# Export to .csv
solution = pd.read_csv('Solution.csv', index_col='TIMESTAMP')
df = pd.DataFrame(index=solution.index)
df['POWER'] = solution.POWER
df['RNN FORECAST'] = y_pred_test 
df.to_csv(path_or_buf='ForecastTemplate3.csv', columns=['RNN FORECAST'], date_format='%Y%m%d %H:%M')




from ann_visualizer.visualize import ann_viz
ann_viz(Regressor, title="title", view= True )

