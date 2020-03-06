import pandas as pd
import tensorflow as tf
from tensorflow import keras
import glob
import datetime
import matplotlib.pyplot as plt
import os


def prepare_train_data(DAY):

    TRAIN_FILES = glob.glob('./data/%s/train_data/*' % DAY, recursive=True)
    TRAIN_DATA = []
    for fl in TRAIN_FILES:
        TRAIN_DATA.append(pd.read_csv(fl))
    df_train = pd.concat(TRAIN_DATA)

    return df_train


def basic_metrics(y_valid, y_proba, DAY, threshold=0):

    df_res = pd.DataFrame({'real':y_valid, 'pred':y_proba[:,0]})

    up_thres = 1+threshold
    down_thres = 1-threshold
    df_res['ok_up'] = ((df_res['real'] > up_thres) & (df_res['pred']>up_thres))
    df_res['ok_down'] = ((df_res['real'] <= down_thres) & (df_res['pred']<=down_thres))

    num_ok_up = df_res['ok_up'].sum()
    num_ok_down = df_res['ok_down'].sum()

    total_up = df_res[(df_res['real']>up_thres)].shape[0]
    total_down = df_res[(df_res['real']<=down_thres)].shape[0]

    total_up_pred = df_res[(df_res['pred']>up_thres)].shape[0]
    total_down_pred = df_res[(df_res['pred']<=down_thres)].shape[0]

    total_recall = (num_ok_up+num_ok_down)/(total_up+total_down)
    total_precision = (num_ok_up+num_ok_down)/(total_up_pred+total_down_pred)

    up_recall = num_ok_up/total_up
    up_precision = num_ok_up/total_up_pred

    down_recall = num_ok_down/total_down
    down_precision = num_ok_down/total_down_pred
    basic_metrics =  pd.DataFrame(
        {
            'total_recall': total_recall,
            'total_precision': total_precision,
            'up_recall': up_recall,
            'up_precision': up_precision,
            'down_recall': down_recall,
            'down_precision': down_precision
        },
        index=[0]
    )
    basic_metrics.to_csv('./data/%s/model/basic_metrics.csv' % DAY)




def train_numeric_model(DAY):

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    X = prepare_train_data(DAY)

    DATE_TRAIN_SPLIT = datetime.datetime.strptime('20170101', '%Y%m%d')
    X['date'] = pd.to_datetime(X['date'])
    X['target'] = X['target']*1000

    X_train = X[X['date'] <= DATE_TRAIN_SPLIT]
    X_valid = X[X['date'] > DATE_TRAIN_SPLIT]

    X_train = X_train[(X_train['target']>0.9*1000) & (X_train['target']<1.1*1000)]


    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop(['target', 'date'], axis=1)
    X_valid = X_valid.drop(['target', 'date'], axis=1)

    INPUT_DIM = X_train.shape[1]
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(100, input_dim=INPUT_DIM, activation='relu'),
      tf.keras.layers.Dense(25, activation='linear'),
      tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(loss="mean_squared_error",
              optimizer=keras.optimizers.Adagrad(learning_rate=0.05),
              metrics=["mean_squared_error"])

    history = model.fit(X_train, y_train, epochs=50,
                    batch_size=32,
                    validation_data=(X_valid, y_valid))

    model.save('./data/%s/model' % DAY)
    model_history = pd.DataFrame(history.history)
    model_history.to_csv('./data/%s/model/model_history.csv' % DAY, index = False)

    plt.figure()
    model_history.plot(figsize=(8, 5))
    plt.grid(True)
    plt.savefig('./data/%s/model/model_history.png' % DAY)


    y_proba = model.predict(X_valid)
    plt.figure()
    plt.hist(y_valid, bins=100, label='Real', alpha=0.6)
    plt.hist(y_proba, bins=100, label='Prediction', alpha=0.6)
    plt.title('Model scoring distribution on validation dataset')
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig('./data/%s/model/distrubution_scoring.png' % DAY)

    basic_metrics(y_valid, y_proba, DAY)
    #model = tf.keras.models.load_model('./data/%s/model/distrubution_scoring.png' % DAY)