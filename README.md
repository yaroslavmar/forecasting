# Forecasting Euro Exchange Rates

Forecasting euro exchange rates

## Execution of pipeline

`python main.py -a import_data`

`python main.py -a train`

`python main.py -a predict`

## Import data 

Data is imported from the European Central Bank via the Quandl API

This step also generates multiple features like rolling average, rolling standard deviations, lags, etc.
The target variable is computed as exchange rate in 7 days over current exchange rate. This is split in three categories:
- return < -0.5%
- return between -0.5% and 0.5%
- return > 0.5%



## Train model

The model consists of a deep and wide neural network. There is a single model that uses the exchange rate of all currencies. 
The categorical variable representing the currency is captures by he 'wide' features in the neural network.


## Generate predictions

