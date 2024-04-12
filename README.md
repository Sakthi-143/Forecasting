# Forecasting 
# Forecasting Assignment      
    
This project involves forecasting the Airlines Passengers data set. The goal is to prepare a document for each model explaining how many dummy variables have been created and the RMSE value for each model. Finally, we will decide which model to use for forecasting.

## Libraries Used    
     
```python
import pandas as pd  
import numpy as np 
from numpy import sqrt 
from pandas import Grouper 
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from pandas.plotting import lag_plot 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
Dataset
The dataset is loaded from an Excel file named ‘Airlines+Data.xlsx’.

Exploratory Data Analysis (EDA)
The EDA process involves checking for null values, duplicated rows, and the data types of the columns. The ‘Month’ column is set as the index of the DataFrame.

Data Visualization
Data visualization techniques used include line plots, histograms, density plots, lag plots, and autocorrelation plots.

Upsampling
The data is upsampled to a monthly frequency using the mean of the ‘Passengers’ column.

# Time Series Analysis

This project involves the analysis of a time series dataset using various techniques such as autocorrelation, partial autocorrelation, sampling, linear interpolation, and time series decomposition.

## Autocorrelation

Autocorrelation is the correlation between a time series (signal) and a delayed version of itself. The Autocorrelation Function (ACF) plots the correlation coefficient against the lag, providing a visual representation of autocorrelation.

## Partial Autocorrelation Function (PACF)

A partial autocorrelation function captures a “direct” correlation between a time series and a lagged version of itself.

```python
import statsmodels.graphics.tsaplots as tsa_plots
with plt.rc_context():
    plt.rc("figure", figsize=(14,6))
    tsa_plots.plot_pacf(df.Sales,lags=20)
    plt.show()
Sampling and Linear Interpolation
The project involves upsampling with respect to the month and then using linear interpolation to fill in the missing values.

upsampled_month = df1.drop(['Quarters','Q1','Q2','Q3','Q4'], axis=1)
upsampled_month = upsampled_month.resample('M').mean()
interpolated_month = upsampled_month.interpolate(method='linear')

Time Series Decomposition
The time series data is decomposed into three components: seasonality, trend, and residuals. Both additive and multiplicative seasonal decompositions are performed.

decompose_ts_add = seasonal_decompose(interpolated_month.Sales, period=12, model='additive')
decompose_ts_add = seasonal_decompose(interpolated_month.Sales, period=12, model='multiplicative')

Train-Test Split
The dataset is split into training and testing sets, with the test data comprising the last 2 years of the time series.

train_data = interpolated_month[:10
