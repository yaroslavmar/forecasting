import glob
import pandas as pd
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt

from train import prepare_train_data

def import_recommendations():
    FILES_RECOMMENDATIONS = glob.glob('./data/20*/recommendations/*', recursive=True)
    RECOMMENDATIONS_DATA = []
    for fl in FILES_RECOMMENDATIONS:
        df_recommendations = pd.read_csv(fl)
        RECOMMENDATIONS_DATA.append(df_recommendations)
    df_recos = pd.concat(RECOMMENDATIONS_DATA)
    return df_recos


df_recos = import_recommendations()
df_recos['date'] = pd.to_datetime(df_recos['date'], format="%Y%m%d")
df_recos['date'] = df_recos['date'].dt.strftime("%Y-%m-%d")


DAY = date.today() - timedelta(1)
DAY = DAY.strftime("%Y%m%d")
df_train = prepare_train_data(DAY)
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['date'] = df_train['date'].dt.strftime("%Y-%m-%d")


df_recos_target = df_recos.merge(df_train[['target', 'date', 'ASSET']],
               left_on=['date', 'ASSET_NAME'],
               right_on=['date', 'ASSET'],
               how='left')

df_recos_target[df_recos_target['class']==2][['target']].describe()
df_recos_target[df_recos_target['class']==2]['target'].sum()

df_recos_target[df_recos_target['class']==0][['target']].describe()
df_recos_target[df_recos_target['class']==0]['target'].sum()


df_recos_target[df_recos_target['class']==2][['target']].describe()
df_recos_target[df_recos_target['class']==2].sort_values('target', ascending= False)[['date', 'ASSET_NAME', 'low','mid', 'high','target']]

df_recos_target[df_recos_target['class']==0][['target']].describe()
df_recos_target[df_recos_target['class']==0].sort_values('target', ascending= False)[['date', 'ASSET_NAME', 'low','mid', 'high','target']]



df_recos_target.plot.scatter(x='high', y='target')
plt.axhline(y=1, linewidth=1, color='r', linestyle='--')
plt.show()


# Import recommendations
# Import train files to get target
# Compute how good recommendations are