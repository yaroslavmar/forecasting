# Forecasting Euro Exchange Rates

Forecasting euro exchange rates

## Execution of pipeline

`python main.py -a import_data`

`python main.py -a train`

`python main.py -a predict`

`python main.py -a recommendations`

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


## Generate recommedations

Recommend a buy/sell acction based on model's predictions. Recommen 'buy' only if proability of increase of rate is much larger than probability of decrease.

## ToDo

- Add sklearn models (logistic regression, random forest, XGBoost)
- Send recommendations to email
- Run univariate data analysis. Compute how good is each individual variable
- Try recurrent neural networks (LSTM)
- When creating recommendations take into account the base model. Do not recommend an asset if model's probabilities are similar to baseline probabilities.
- Backtesting
