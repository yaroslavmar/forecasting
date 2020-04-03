

from datetime import date, timedelta, datetime
import argparse
import pandas as pd

import import_data
from list_of_assets import ALL_ASSETS
import train



parser = argparse.ArgumentParser()
parser.add_argument("-a", "--action", required=True)
parser.add_argument("-d", "--day", required=False)
args = parser.parse_args().__dict__
action = args['action']

if __name__ == '__main__':

    if args['day'] == None:
        DAY = date.today() - timedelta(1)
        DAY = DAY.strftime("%Y%m%d")
    else:
        DAY = datetime.strptime(args['day'], "%Y%m%d").date().strftime("%Y%m%d")

    if action == 'import_data':
        for ASSET in ALL_ASSETS:
            ASSET_NAME = ASSET.replace('/', '_')
            print('Importing asset', ASSET)
            import_data.import_data(ASSET)
            import_data.create_train_predict(ASSET_NAME, DAY)

    if action == 'train':
        train.baseline_model(DAY)
        train.train_model(DAY)


    if action == 'predict':
        for ASSET in ALL_ASSETS:
            print('Prediction for asset', ASSET)
            ASSET_NAME = ASSET.replace('/', '_')
            train.generate_prediction(DAY, ASSET_NAME)

    if action == 'recommendations':
        train.get_recommendations(DAY)

    if action == 'backfill_train':
        # Run on first day of the month!
        DAYS_BACKFILL = pd.date_range('2019-01-01', DAY, freq='1M')-pd.offsets.MonthBegin(1)
        DAYS_BACKFILL = [x.strftime("%Y%m%d") for x in DAYS_BACKFILL]
        for DAY in DAYS_BACKFILL:
            print('Backfilling day', DAY)
            # IMPORT DATA
            print('Import data')
            for ASSET in ALL_ASSETS:
                ASSET_NAME = ASSET.replace('/', '_')
                #import_data.import_data(ASSET)
                import_data.create_train_predict(ASSET_NAME, DAY)
            # TRAIN MODELS
            print('Train models')
            train.baseline_model(DAY)
            train.train_model(DAY)

            if action == 'backfill_train':
                # Run on Mondays!
                DAYS_BACKFILL = pd.date_range('2019-01-01', DAY, freq='1M') - pd.offsets.MonthBegin(1)
                DAYS_BACKFILL = pd.date_range(start='2019-01-01', end=DAY,
                         freq='W-MON').strftime("%Y%m%d").tolist()
                for DAY in DAYS_BACKFILL:
                    print('Predictions')
                    for ASSET in ALL_ASSETS:
                        ASSET_NAME = ASSET.replace('/', '_')
                        train.generate_prediction(DAY, ASSET_NAME)
                    # RECOMMENDATIONS
                    print('Recommendations')
                    train.get_recommendations(DAY)
