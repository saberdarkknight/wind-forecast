# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:51:02 2018

@author: Lancer
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from math import sqrt


sns.set(style="white")


# Import data
train_data = pd.read_csv('TrainData.csv', usecols=['TIMESTAMP', 'POWER'], parse_dates=[0], index_col='TIMESTAMP')
solution = pd.read_csv('Solution.csv', parse_dates=[0], index_col='TIMESTAMP')

plot_acf(train_data.POWER, lags=50)


history = [x for x in train_data.POWER]
test = [x for x in solution.POWER]
predictions = list()

for t in range(len(solution.POWER)):
    model = ARIMA(history, order=(5,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0][0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    finished = (t/len(solution.POWER))*100
    print('predicted=%f, expected=%f, percentage=%f%%' % (yhat, obs, finished))
    
    
error = mean_squared_error(test, predictions)
RMSE = sqrt(error)
print('Test RMSE: %.3f' % RMSE)


'''
params = {'legend.fontsize': 40}
plt.rcParams.update(params)
df = pd.DataFrame(index=solution.index)
df['POWER'] = solution.POWER
df['ARIMA FORECAST'] = predictions
styles = ['-','--']
fig, ax = plt.subplots(figsize=(12,8))
df.plot(style=styles, ax=ax, x_compat=True)
ax.set_title('Real & Predicted Wind Power (ARIMA)')
ax.set_xlabel('Time')
ax.set_ylabel('Power (normalized)')
plt.savefig('Exercise 3 - Forecast.png', dpi = 300)
'''


import matplotlib
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

params = {'legend.fontsize': 40}
plt.rcParams.update(params)
x=np.arange(720)
fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(x,solution,c='b',label='observed',  linewidth=6.0)
ax.plot(x,predictions,c='r',label='LR predicted',  linewidth=6.0)
ax.xaxis.set_ticks([i*60 for i in range(0,13)])
ax.set_ylabel('Power Generation', fontsize=44)
ax.set_xlabel('Time', fontsize=44)
ax.tick_params(labelsize=44)
ax.grid()
plt.legend(loc=2)
#plt.title('Real & Predicted Wind Power')
ax.tick_params(labelsize=44)
plt.draw()



# Export to .csv
#df.to_csv(path_or_buf='ForecastTemplate3.csv', columns=['ARIMA FORECAST'], date_format='%Y%m%d %H:%M')