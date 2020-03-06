
import pandas as pd
import datetime
import quandl
from keys import QUANDL_KEY
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

    cols_train = [x for x in data_raw.columns if 'lag' in x]
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

