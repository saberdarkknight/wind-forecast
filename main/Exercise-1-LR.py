# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('TrainData.csv')
X_train = dataset_train.iloc[:, 4].values
X_train= X_train.reshape(-1, 1)

y_train = dataset_train.iloc[:, 1].values
y_train= y_train.reshape(-1, 1)

dataset_test_x = pd.read_csv('WeatherForecastInput.csv')
X_test = dataset_test_x.iloc[:, 3].values
X_test= X_test.reshape(-1, 1)

dataset_test_y =pd.read_csv('Solution.csv')
y_test= dataset_test_y.iloc[:, 1].values
y_test= y_test.reshape(-1, 1)


# Fitting the Regression Model to the dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)
# Predicting a new result
y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)


import math 
#Root mean square error
error_test = y_test-y_pred_test
error_test_sq=np.square (error_test)
RMSE_test=math.sqrt(sum(error_test_sq)/error_test.shape[0])
 

error_train = y_train-y_pred_train
error_train_sq=np.square (error_train)
RMSE_train=math.sqrt(sum(error_train_sq)/error_train.shape[0])

Overfitting_indicator_test =RMSE_train/RMSE_test

result=([['RMSE_Train','RMSE_Test_set','Overfitting_test'],
         [ RMSE_train, RMSE_test ,Overfitting_indicator_test]])
result=np.array(result)


#Efficiency index
Yav_test=(1/error_test.shape[0])*sum(y_test)
Yav_test=float(Yav_test)
ST_test=sum((y_test-Yav_test)**2)
ST_test=float(ST_test)
EI_test= (ST_test-sum(error_test_sq))/ST_test
EI_test=float(EI_test)



Yav_train=(1/error_train.shape[0])*sum(y_train)
Yav_train=float(Yav_train)
ST_train=sum((y_train-Yav_train)**2)
ST_train=float(ST_train)
EI_train= (ST_train-sum(error_train_sq))/ST_train
EI_train=float(EI_train)

Performance_parameters=([['.','Train_set','Test_set'],
                         ['RMSE', RMSE_train, RMSE_test],
                         ['Mean',Yav_train,Yav_test],
                        ['EI',EI_train,EI_test]])
Performance_parameters=np.array(Performance_parameters)


import matplotlib
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
ax.set_ylabel('Power Generation', fontsize=44)
ax.set_xlabel('Time', fontsize=44)
ax.tick_params(labelsize=36)
ax.grid()
plt.legend(loc=2)
plt.title('Linear Regression', fontsize=30)
plt.draw()


# Export to .csv
solution = pd.read_csv('Solution.csv', index_col='TIMESTAMP')
df = pd.DataFrame(index=dataset_test_y.index)
df['POWER'] = solution.POWER
df['LR FORECAST'] = y_pred_test 
df.to_csv(path_or_buf='ForecastTemplate1.csv', columns=['LR FORECAST'], date_format='%Y%m%d %H:%M')




