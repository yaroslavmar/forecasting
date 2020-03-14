
import pandas as pd
import datetime
import quandl
from api_keys import QUANDL_KEY
quandl.ApiConfig.api_key = QUANDL_KEY
import yaml


import utils


stream = open('config.yaml', 'r')
config = yaml.load(stream, yaml.Loader)

def import_data(ASSET):
    df = quandl.get(ASSET)
    ASSET_NAME = ASSET.replace('/', '_')
    num_lags = list(range(1, 30, 1))
    num_leads = list(range(1, 10, 1))
    COLUMN_VALUE = 'Value'

    df_var = add_relative_values(df, COLUMN_VALUE, num_lags, type_value='lag')
    df_var = add_relative_values(df_var, COLUMN_VALUE, num_leads, type_value='lead')
    df_var = df_var.sort_index(ascending=True)
    df_var['date'] = df_var.index

    df_var['rolling_mean_7'] = df_var['Value'].rolling(7).mean()/df_var['Value']
    df_var['rolling_mean_14'] = df_var['Value'].rolling(14).mean()/df_var['Value']
    df_var['rolling_mean_21'] = df_var['Value'].rolling(21).mean()/df_var['Value']
    df_var['rolling_mean_28'] = df_var['Value'].rolling(28).mean()/df_var['Value']
    df_var['rolling_mean_35'] = df_var['Value'].rolling(35).mean()/df_var['Value']
    df_var['rolling_mean_60'] = df_var['Value'].rolling(60).mean()/df_var['Value']

    df_var['rolling_mean_7_shift_7']=df_var['rolling_mean_7'].shift(7)
    df_var['rolling_mean_14_shift_7'] = df_var['rolling_mean_14'].shift(7)
    df_var['rolling_mean_21_shift_7'] = df_var['rolling_mean_21'].shift(7)
    df_var['rolling_mean_28_shift_7'] = df_var['rolling_mean_28'].shift(7)
    df_var['rolling_mean_35_shift_7'] = df_var['rolling_mean_35'].shift(7)

    df_var['rolling_mean_7_shift_14'] = df_var['rolling_mean_7'].shift(14)
    df_var['rolling_mean_14_shift_14'] = df_var['rolling_mean_14'].shift(14)
    df_var['rolling_mean_21_shift_14'] = df_var['rolling_mean_21'].shift(14)
    df_var['rolling_mean_28_shift_14'] = df_var['rolling_mean_28'].shift(14)
    df_var['rolling_mean_35_shift_14'] = df_var['rolling_mean_35'].shift(14)

    df_var['rolling_mean_7_shift_21'] = df_var['rolling_mean_7'].shift(21)
    df_var['rolling_mean_14_shift_21'] = df_var['rolling_mean_14'].shift(21)
    df_var['rolling_mean_21_shift_21'] = df_var['rolling_mean_21'].shift(21)
    df_var['rolling_mean_28_shift_21'] = df_var['rolling_mean_28'].shift(21)
    df_var['rolling_mean_35_shift_21'] = df_var['rolling_mean_35'].shift(21)

    df_var['rolling_std_7'] = df_var['Value'].rolling(7).std()/df_var['Value']
    df_var['rolling_std_14'] = df_var['Value'].rolling(14).std()/df_var['Value']
    df_var['rolling_std_21'] = df_var['Value'].rolling(21).std()/df_var['Value']
    df_var['rolling_std_28'] = df_var['Value'].rolling(28).std()/df_var['Value']
    df_var['rolling_std_35'] = df_var['Value'].rolling(35).std()/df_var['Value']
    df_var['rolling_std_60'] = df_var['Value'].rolling(60).std()/df_var['Value']

    df_var['rolling_std_7_shift_7']=df_var['rolling_std_7'].shift(7)
    df_var['rolling_std_14_shift_7'] = df_var['rolling_std_14'].shift(7)
    df_var['rolling_std_21_shift_7'] = df_var['rolling_std_21'].shift(7)
    df_var['rolling_std_28_shift_7'] = df_var['rolling_std_28'].shift(7)
    df_var['rolling_std_35_shift_7'] = df_var['rolling_std_35'].shift(7)

    df_var['rolling_std_7_shift_14']=df_var['rolling_std_7'].shift(14)
    df_var['rolling_std_14_shift_14'] = df_var['rolling_std_14'].shift(14)
    df_var['rolling_std_21_shift_14'] = df_var['rolling_std_21'].shift(14)
    df_var['rolling_std_28_shift_14'] = df_var['rolling_std_28'].shift(14)
    df_var['rolling_std_35_shift_14'] = df_var['rolling_std_35'].shift(14)

    df_var['rolling_std_7_shift_21']=df_var['rolling_std_7'].shift(21)
    df_var['rolling_std_14_shift_21'] = df_var['rolling_std_14'].shift(21)
    df_var['rolling_std_21_shift_21'] = df_var['rolling_std_21'].shift(21)
    df_var['rolling_std_28_shift_21'] = df_var['rolling_std_28'].shift(21)
    df_var['rolling_std_35_shift_21'] = df_var['rolling_std_35'].shift(21)


    df_var['rolling_max_7'] = df_var['Value'].rolling(7).max()/df_var['Value']
    df_var['rolling_max_14'] = df_var['Value'].rolling(14).max()/df_var['Value']
    df_var['rolling_max_21'] = df_var['Value'].rolling(21).max()/df_var['Value']
    df_var['rolling_max_28'] = df_var['Value'].rolling(28).max()/df_var['Value']
    df_var['rolling_max_35'] = df_var['Value'].rolling(35).max()/df_var['Value']
    df_var['rolling_max_60'] = df_var['Value'].rolling(60).max()/df_var['Value']

    df_var['rolling_max_7_shift_7']=df_var['rolling_max_7'].shift(7)
    df_var['rolling_max_14_shift_7'] = df_var['rolling_max_14'].shift(7)
    df_var['rolling_max_21_shift_7'] = df_var['rolling_max_21'].shift(7)
    df_var['rolling_max_28_shift_7'] = df_var['rolling_max_28'].shift(7)
    df_var['rolling_max_35_shift_7'] = df_var['rolling_max_35'].shift(7)

    df_var['rolling_max_7_shift_14']=df_var['rolling_max_7'].shift(14)
    df_var['rolling_max_14_shift_14'] = df_var['rolling_max_14'].shift(14)
    df_var['rolling_max_21_shift_14'] = df_var['rolling_max_21'].shift(14)
    df_var['rolling_max_28_shift_14'] = df_var['rolling_max_28'].shift(14)
    df_var['rolling_max_35_shift_14'] = df_var['rolling_max_35'].shift(14)

    df_var['rolling_max_7_shift_21']=df_var['rolling_max_7'].shift(21)
    df_var['rolling_max_14_shift_21'] = df_var['rolling_max_14'].shift(21)
    df_var['rolling_max_21_shift_21'] = df_var['rolling_max_21'].shift(21)
    df_var['rolling_max_28_shift_21'] = df_var['rolling_max_28'].shift(21)
    df_var['rolling_max_35_shift_21'] = df_var['rolling_max_35'].shift(21)

    df_var['rolling_min_7'] = df_var['Value'].rolling(7).min()/df_var['Value']
    df_var['rolling_min_14'] = df_var['Value'].rolling(14).min()/df_var['Value']
    df_var['rolling_min_21'] = df_var['Value'].rolling(21).min()/df_var['Value']
    df_var['rolling_min_28'] = df_var['Value'].rolling(28).min()/df_var['Value']
    df_var['rolling_min_35'] = df_var['Value'].rolling(35).min()/df_var['Value']
    df_var['rolling_min_60'] = df_var['Value'].rolling(60).min()/df_var['Value']

    df_var['rolling_min_7_shift_7']=df_var['rolling_min_7'].shift(7)
    df_var['rolling_min_14_shift_7'] = df_var['rolling_min_14'].shift(7)
    df_var['rolling_min_21_shift_7'] = df_var['rolling_min_21'].shift(7)
    df_var['rolling_min_28_shift_7'] = df_var['rolling_min_28'].shift(7)
    df_var['rolling_min_35_shift_7'] = df_var['rolling_min_35'].shift(7)

    df_var['rolling_min_7_shift_14']=df_var['rolling_min_7'].shift(14)
    df_var['rolling_min_14_shift_14'] = df_var['rolling_min_14'].shift(14)
    df_var['rolling_min_21_shift_14'] = df_var['rolling_min_21'].shift(14)
    df_var['rolling_min_28_shift_14'] = df_var['rolling_min_28'].shift(14)
    df_var['rolling_min_35_shift_14'] = df_var['rolling_min_35'].shift(14)

    df_var['rolling_min_7_shift_21']=df_var['rolling_min_7'].shift(21)
    df_var['rolling_min_14_shift_21'] = df_var['rolling_min_14'].shift(21)
    df_var['rolling_min_21_shift_21'] = df_var['rolling_min_21'].shift(21)
    df_var['rolling_min_28_shift_21'] = df_var['rolling_min_28'].shift(21)
    df_var['rolling_min_35_shift_21'] = df_var['rolling_min_35'].shift(21)


    df_var.to_csv('./data/raw_data/raw_data_%s.csv' % ASSET_NAME, index=False)



def add_relative_values(df, TARGET_COLUMN, index_lags, type_value):
    if type_value == 'lag':
        df = df.sort_index(ascending=True)
    if type_value == 'lead':
        df = df.sort_index(ascending=False)

    for i in index_lags:
        df['%s_%s' % (type_value, i)] = df[TARGET_COLUMN].shift(i)
        df['rel_%s_%s' % (type_value, i)] = df[TARGET_COLUMN] / df['%s_%s' % (type_value, i)]
        df.drop(['%s_%s' % (type_value, i)], axis=1, inplace=True)
    return df


def create_train_predict(ASSET_NAME, DAY):
    data_raw = pd.read_csv('./data/raw_data/raw_data_%s.csv' % ASSET_NAME)
    data_raw['date'] = pd.to_datetime(data_raw['date'])
    DAY_FORMAT = datetime.datetime.strptime(DAY,'%Y%m%d')
    data_raw = data_raw[data_raw['date'] <= DAY_FORMAT]

    cols_train = [x for x in data_raw.columns if 'lag' in x or 'rolling' in x]
    col_target = config['target_column']

    data_raw = data_raw.sort_values('date', ascending=True)
    df_train = data_raw[cols_train + [col_target, 'date']].dropna()
    df_train = df_train.rename(columns={col_target: 'target'})

    df_predict = data_raw[cols_train + ['date']].tail(1)

    PATH_TRAIN = './data/%s/train_data/train_%s.csv' % (DAY, ASSET_NAME)
    utils.create_if_necessary(PATH_TRAIN)
    df_train.to_csv(PATH_TRAIN, index=False)

    PATH_PREDICT = './data/%s/predict_data/predict_%s.csv' % (DAY, ASSET_NAME)
    utils.create_if_necessary(PATH_PREDICT)
    df_predict.to_csv(PATH_PREDICT, index=False)

