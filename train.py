import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import glob
import datetime
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.metrics import classification_report


def prepare_train_data(DAY):

    TRAIN_FILES = glob.glob('./data/%s/train_data/*' % DAY, recursive=True)
    TRAIN_DATA = []
    for fl in TRAIN_FILES:
        df_asset = pd.read_csv(fl)
        ASSET_NAME = os.path.basename(fl).replace('train_', '').replace('.csv', '')
        df_asset['ASSET'] = ASSET_NAME
        TRAIN_DATA.append(df_asset)
    df_train = pd.concat(TRAIN_DATA)

    return df_train



def train_model(DAY):

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    X = prepare_train_data(DAY)

    # REMOVE OUTLIERS IN TARGET
    X = X[(X['target']>0.9) & (X['target']<1.1)]

    # SPLIT TARGET INTO CATEGORIES
    X['target'] = pd.cut(X['target'],[-10000,0.995, 1.005, 10000], labels=[0, 1, 2])

    # TRAIN-VALIDATION SPLIT
    DATE_TRAIN_SPLIT = datetime.datetime.strptime('20170101', '%Y%m%d')
    X['date'] = pd.to_datetime(X['date'])
    X_train = X[X['date'] <= DATE_TRAIN_SPLIT]
    X_valid = X[X['date'] > DATE_TRAIN_SPLIT]

    # ONE HOT ENCODING OF ASSETS
    categories = np.array(list(set(X['ASSET'].astype(str).values))).reshape(-1, 1)
    ohe = preprocessing.OneHotEncoder()
    ohe.fit(categories)
    X_wide_train = ohe.transform(np.array(X_train['ASSET']).reshape(-1, 1))
    X_wide_valid = ohe.transform(np.array(X_valid['ASSET']).reshape(-1, 1))

    y_train = X_train['target']
    y_valid = X_valid['target']
    X_train = X_train.drop(['target', 'date', 'ASSET'], axis=1)
    X_valid = X_valid.drop(['target', 'date', 'ASSET'], axis=1)

    labels_train = to_categorical(y_train)
    labels_valid = to_categorical(y_valid)

    # WIDE AND DEEP NEURAL NETWORK
    INPUT_WIDE = X_wide_train.shape[1]
    INPUT_DEEP = X_train.shape[1]
    input_A = keras.layers.Input(shape=[INPUT_WIDE], name="wide_input")
    input_B = keras.layers.Input(shape=[INPUT_DEEP], name="deep_input")
    hidden1 = keras.layers.Dense(500, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(350, activation="relu")(hidden1)
    hidden3 = keras.layers.Dense(250, activation="relu")(hidden2)
    hidden4 = keras.layers.Dense(150, activation="relu")(hidden3)
    hidden5 = keras.layers.Dense(100, activation="relu")(hidden4)
    concat = keras.layers.concatenate([input_A, hidden5])
    output = keras.layers.Dense(3, activation='softmax', name="output")(concat)
    model = keras.Model(inputs=[input_A, input_B], outputs=[output])

    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(0.01),
                  metrics=["categorical_accuracy"])

    history = model.fit((X_wide_train, X_train), labels_train, epochs=50,
                    batch_size=32,
                    validation_data=((X_wide_valid, X_valid), labels_valid))

    model.save('./data/%s/model/model' % DAY)

    filehandler = open('./data/%s/model/one_hot_encoder.pkl' % DAY, 'wb')
    pickle.dump(ohe, filehandler)

    # filehandler = open('./data/%s/model/one_hot_encoder.pkl' % DAY, 'rb')
    # x = pickle.load(filehandler)
    model_history = pd.DataFrame(history.history)
    model_history.to_csv('./data/%s/model/model_history.csv' % DAY, index = False)

    plt.figure()
    model_history.plot(figsize=(8, 5))
    plt.grid(True)
    plt.savefig('./data/%s/model/model_history.png' % DAY)

    preds = model.predict((X_wide_valid, X_valid))
    y_classes = preds.argmax(axis=-1)
    np.unique(y_classes, return_counts=True)

    plt.figure()
    plt.hist(preds[:, 0], bins=50, alpha=0.5, label='Class 0')
    plt.hist(preds[:, 1], bins=50, alpha=0.5, label='Class 1')
    plt.hist(preds[:, 2], bins=50, alpha=0.5, label='Class 2')
    plt.title('Distribution of predicted probabilities')
    plt.legend()
    plt.grid()
    plt.savefig('./data/%s/model/probabilities_distribution.png' % DAY)

    y_pred = np.argmax(preds, axis=1)
    target_names = ['low', 'mid', 'high']
    model_report = classification_report(y_valid, y_pred, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(model_report).transpose()
    df_report = round(df_report, 4)
    df_report.to_csv('./data/%s/model/model_report.csv' % DAY)



def generate_prediction(DAY, ASSET_NAME):
    # Import predict data
    PATH_PREDICT = './data/%s/predict_data/predict_%s.csv' % (DAY, ASSET_NAME)
    df_predict = pd.read_csv(PATH_PREDICT)
    print(PATH_PREDICT)

    # Import model
    # Import one hot encoder
    filehandler = open('./data/%s/model/one_hot_encoder.pkl' % DAY, 'rb')
    ohe = pickle.load(filehandler)
    X_wide_predict = ohe.transform(np.array(df_predict['ASSET']).reshape(-1, 1))
    # Predict
    # Save
