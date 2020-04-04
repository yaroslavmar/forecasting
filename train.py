import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import datetime
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.metrics import classification_report, log_loss
import utils
import glob

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

def get_train_test_split_date(X):
    DAYS_RANGE = round((X.date.max()- X.date.min()).days*0.9)
    DATE_TRAIN_SPLIT = (X.date.min() + datetime.timedelta(days=DAYS_RANGE))
    return DATE_TRAIN_SPLIT


def train_model(DAY):

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    X = prepare_train_data(DAY)

    # REMOVE OUTLIERS IN TARGET
    X = X[(X['target']>0.9) & (X['target']<1.1)]

    # SPLIT TARGET INTO CATEGORIES
    X['target'] = pd.cut(X['target'],[-10000,0.995, 1.005, 10000], labels=[0, 1, 2])

    # TRAIN-VALIDATION SPLIT
    X['date'] = pd.to_datetime(X['date'])
    DATE_TRAIN_SPLIT = get_train_test_split_date(X)
    print('Train test date split:', DATE_TRAIN_SPLIT)
    print('Max date in train', X.date.max())

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
    hidden1 = keras.layers.Dense(15, activation=keras.layers.ELU())(input_B)
    hidden2 = keras.layers.Dense(10, activation=keras.layers.ELU())(hidden1)
    hidden3 = keras.layers.Dense(5, activation=keras.layers.ELU())(hidden2)
    #hidden4 = keras.layers.Dense(40, activation=keras.layers.ELU())(hidden3)
    #hidden5 = keras.layers.Dense(20, activation=keras.layers.ELU())(hidden4)
    concat = keras.layers.concatenate([input_A, hidden3])
    output = keras.layers.Dense(3, activation='softmax', name="output")(concat)
    model = keras.Model(inputs=[input_A, input_B], outputs=[output])

    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(0.001),
                  metrics=["categorical_accuracy"])

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    history = model.fit((X_wide_train, X_train), labels_train, callbacks=[callback],
                        epochs=100,
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

def baseline_model(DAY):

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    X = prepare_train_data(DAY)

    # REMOVE OUTLIERS IN TARGET
    X = X[(X['target']>0.9) & (X['target']<1.1)]

    # SPLIT TARGET INTO CATEGORIES
    X['target'] = pd.cut(X['target'],[-10000,0.995, 1.005, 10000], labels=[0, 1, 2])

    # TRAIN-VALIDATION SPLIT
    X['date'] = pd.to_datetime(X['date'])
    DATE_TRAIN_SPLIT = get_train_test_split_date(X)
    #DATE_TRAIN_SPLIT = datetime.datetime.strptime('20170101', '%Y%m%d')
    X_train = X[X['date'] <= DATE_TRAIN_SPLIT]
    X_valid = X[X['date'] > DATE_TRAIN_SPLIT]

    # BASELINE BY ASSET
    def compute_baseline_score(df_hist):
        base_preds = df_hist.groupby('ASSET')['target'].value_counts()
        baseline_assets = base_preds / base_preds.groupby(level=0).sum()
        baseline_assets = baseline_assets.reset_index(level=0, drop=False)
        baseline_assets['class'] = baseline_assets.index
        baseline_assets_preds = baseline_assets.pivot(index='ASSET', columns='class', values='target')
        baseline_assets_preds = baseline_assets_preds.reset_index()
        baseline_assets_preds.columns = ['ASSET_NAME', 'low_baseline', 'mid_baseline', 'high_baseline']
        return baseline_assets_preds

    # PREDICTIONS USING LAST DATA
    DAY_FORMAT = datetime.datetime.strptime(DAY, '%Y%m%d')
    DAY_START_BASELINE = DAY_FORMAT - datetime.timedelta(180)
    baseline_assets_preds = compute_baseline_score(X[X['date'] > DAY_START_BASELINE])
    baseline_assets_preds = baseline_assets_preds.fillna(0)
    PATH_BASELINE_MODEL_PREDICTIONS = './data/%s/model/baseline_model_predictions.csv' % DAY
    utils.create_if_necessary(PATH_BASELINE_MODEL_PREDICTIONS)
    baseline_assets_preds.to_csv(PATH_BASELINE_MODEL_PREDICTIONS, index=False)

    # PREDICTIONS WITH TRAIN DATA TO EVALUATE ON VALIDATION DATA
    DAY_START_BASELINE_TRAIN = DATE_TRAIN_SPLIT - datetime.timedelta(180)
    baseline_assets_train = compute_baseline_score(X_train[X_train['date'] > DAY_START_BASELINE_TRAIN])
    baseline_assets_train = baseline_assets_train.fillna(0)
    baseline_assets_train.to_csv('./data/%s/model/baseline_model_train_predictions.csv' % DAY, index=False)

    y_valid = X_valid['target']

    X_pred = X_valid[['ASSET']]
    X_pred.columns = ['ASSET_NAME']
    preds = X_pred.merge(baseline_assets_preds, how='left', on='ASSET_NAME')
    preds = preds.drop('ASSET_NAME', axis=1)
    preds = np.array(preds)

    y_pred = np.argmax(preds, axis=1)
    target_names = ['low', 'mid', 'high']
    model_report = classification_report(y_valid, y_pred, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(model_report).transpose()
    df_report = round(df_report, 4)
    df_report.to_csv('./data/%s/model/baseline_model_report.csv' % DAY)

    log_loss_baseline = log_loss(y_valid, preds)
    file_ll = open('./data/%s/model/log_loss_baseline.csv' % DAY, "w")
    file_ll.write(str(log_loss_baseline))
    file_ll.close()



def get_latest_model(DAY):
    execs_models = glob.glob('./data/20*/model/*', recursive=True)
    execs_models = [int(x.split('/')[2]) for x in execs_models]
    execs_models = np.array(execs_models)
    execs_models = np.unique(execs_models)
    execs_models = execs_models[execs_models <= int(DAY)]
    max_execution = execs_models.max()
    return max_execution

def generate_prediction(DAY, ASSET_NAME):

    # IMPORT PREDICTION DATA
    PATH_PREDICT = './data/%s/predict_data/predict_%s.csv' % (DAY, ASSET_NAME)
    df_predict = pd.read_csv(PATH_PREDICT)
    df_predict['ASSET'] = ASSET_NAME

    # IMPORT MODEL AND ONE HOT ENCODER
    DAY_MODEL = get_latest_model(DAY)
    # print('Model was trained on:', DAY_MODEL)

    filehandler = open('./data/%s/model/one_hot_encoder.pkl' % DAY_MODEL, 'rb')
    ohe = pickle.load(filehandler)
    X_wide_predict = ohe.transform(np.array(df_predict['ASSET']).reshape(-1, 1))

    model = keras.models.load_model('./data/%s/model/model' % DAY_MODEL)

    # CREATE PREDICTION
    df_predict = df_predict.drop(['date', 'ASSET'], axis=1)
    preds = model.predict((X_wide_predict, df_predict))
    y_class = preds.argmax(axis=-1)[0]
    df_preds = pd.DataFrame(preds)
    df_preds.columns = ['low', 'mid', 'high']
    df_preds['class'] = y_class
    df_preds.insert(0, 'ASSET_NAME', ASSET_NAME)
    df_preds.insert(0, 'date', DAY)

    # SAVE PREDICTIONS
    PATH_PREDICTIONS= './data/%s/predictions/prediction_%s.csv' % (DAY, ASSET_NAME)
    utils.create_if_necessary(PATH_PREDICTIONS)
    df_preds.to_csv(PATH_PREDICTIONS, index=False)


def get_recommendations(DAY):

    # Model predictions
    PATH_PREDICTIONS = './data/%s/predictions/*' % (DAY)
    PATH_FILES_PREDICTIONS = glob.glob(PATH_PREDICTIONS, recursive=True)

    PREDICTIONS = []
    for fl in PATH_FILES_PREDICTIONS:
        df_asset_preds = pd.read_csv(fl)
        PREDICTIONS.append(df_asset_preds)
    df_predictions = pd.concat(PREDICTIONS)

    # Baseline predictions
    DAY_MODEL = get_latest_model(DAY)
    baseline_pred = pd.read_csv('./data/%s/model/baseline_model_predictions.csv' % DAY_MODEL)

    # Generate recommendations
    df_predictions = df_predictions.merge(baseline_pred, on='ASSET_NAME', how='left')

    df_predictions['ratio_high_low'] = df_predictions['high']/df_predictions['low']
    df_predictions['difference_high_low'] = (df_predictions['high'] - df_predictions['low']).abs()
    df_predictions['difference_low_baseline'] = df_predictions['low'] - df_predictions['low_baseline']
    df_predictions['difference_high_baseline'] = df_predictions['high'] - df_predictions['high_baseline']

    df_recommendations = df_predictions[df_predictions['class'] != 1]
    df_recommendations = df_recommendations[df_recommendations['difference_high_low'] > 0.05]
    df_recommendations = df_recommendations[df_recommendations['difference_low_baseline'] > 0.05]
    df_recommendations = df_recommendations[df_recommendations['difference_high_baseline'] > 0.05]

    PATH_RECOMMENDATIONS = './data/%s/recommendations/recommendations.csv' % (DAY)
    utils.create_if_necessary(PATH_RECOMMENDATIONS)
    df_recommendations.to_csv(PATH_RECOMMENDATIONS, index=False)

    pd.set_option('display.max_columns', 500)
    print(df_recommendations)

