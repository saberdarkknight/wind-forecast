# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt


# Set plot styling
sns.set(style="white")
#sns.choose_colorbrewer_palette("sequential")

# Import data
train_data = pd.read_csv('TrainData.csv', usecols=['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10'], index_col='TIMESTAMP')
weather_forecast = pd.read_csv('WeatherForecastInput.csv', usecols=['TIMESTAMP', 'U10', 'V10', 'WS10'], index_col='TIMESTAMP')
solution = pd.read_csv('Solution.csv', index_col='TIMESTAMP')


dataset_test_y =pd.read_csv('Solution.csv')
y_test= dataset_test_y.iloc[:, 1].values
y_test= y_test.reshape(-1, 1)


# Change index from string to date-time format
train_data.index = pd.to_datetime(train_data.index)
weather_forecast.index = pd.to_datetime(weather_forecast.index)
solution.index = pd.to_datetime(solution.index)


# Create angle (radians) from zonal and meridional components
train_data['ANGLE'] = np.arctan2(train_data['V10'], train_data['U10'])
weather_forecast['ANGLE'] = np.arctan2(weather_forecast['V10'], weather_forecast['U10'])

# Change from -pi - +pi to 0pi - 2pi 
train_data['ANGLE'] = (train_data['ANGLE'] + 2*np.pi) % (2*np.pi)
weather_forecast['ANGLE'] = (weather_forecast['ANGLE'] + 2*np.pi) % (2*np.pi)

# Plotting pair grid
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
g = sns.PairGrid(train_data)
g.map(plt.scatter);

# Train Multiple Linear Regression model
MLreg = linear_model.LinearRegression()
X = train_data.as_matrix(columns=['WS10','ANGLE'])
y = train_data['POWER']
MLreg.fit(X, y)

# The coefficients
print('Coefficients: \n', MLreg.coef_)


# Train Linear Regression model
Lreg = linear_model.LinearRegression()
X = train_data['WS10'].values.reshape(-1, 1)
y = train_data['POWER']
Lreg.fit(X, y)

# The coefficients
print('Coefficients: \n', Lreg.coef_)



# Multiple Linear Regression prediction from weather forecast(speed and angle)
MLRpower_prediction = MLreg.predict(weather_forecast[['WS10','ANGLE']]);

# The mean squared error
print("Root mean squared error: %.3f"
      % sqrt(mean_squared_error(solution.POWER, MLRpower_prediction)))



# Linear Regression prediction from weather forecast(speed)
LRpower_prediction = Lreg.predict(weather_forecast['WS10'].values.reshape(-1, 1));

# The mean squared error
print("Root mean squared error: %.3f"
      % sqrt(mean_squared_error(solution.POWER, LRpower_prediction)))


'''
# Plot time series of real and predicted wind power
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

params = {'legend.fontsize': 30}
plt.rcParams.update(params)
df = pd.DataFrame(index=solution.index)
df['POWER'] = solution.POWER
df['MLR FORECAST'] = MLRpower_prediction
df['LR FORECAST'] = LRpower_prediction
styles = ['-','--','--']
fig, ax = plt.subplots(figsize=(12,8))
df.plot(style=styles, ax=ax, x_compat=True, linewidth=6.0, fontsize=36)
ax.set_title('Real & Predicted Wind Power (MLR)')
ax.set_ylabel('Power Generation', fontsize=36)
ax.set_xlabel('Time', fontsize=36)
ax.grid()
plt.savefig('Exercise 2 - Forecast.png', dpi = 300)


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
ax.plot(x,y_test[:],c='b',label='observed',  linewidth=6.0)
ax.plot(x,LRpower_prediction,c='r',label='LR predicted',  linewidth=6.0)
ax.plot(x,MLRpower_prediction,c='y',label='MLR predicted', linewidth=6.0)
ax.xaxis.set_ticks([i*60 for i in range(0,13)])
ax.set_ylabel('Power Generation', fontsize=44)
ax.set_xlabel('Time', fontsize=44)
ax.tick_params(labelsize=44)
ax.grid()
plt.legend(loc=2)
#plt.title('Real & Predicted Wind Power')
plt.draw()



# Export to .csv
#df.to_csv(path_or_buf='ForecastTemplate2.csv', columns=['MLR FORECAST'], date_format='%Y%m%d %H:%M')

