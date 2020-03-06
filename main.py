

from datetime import date, timedelta
import import_data

import train

if __name__ == '__main__':

    DAY = date.today() - timedelta(days=1)
    DAY = DAY.strftime("%Y%m%d")
    train.train_numeric_model(DAY)
    ALL_ASSETS = [
        'ECB/EURPLN',
        'ECB/EURGBP',
        'ECB/EURIDR',
        'ECB/EURKRW',
        'ECB/EURSEK',
        'ECB/EURHKD',
        'ECB/EURRON',
        'ECB/EURCHF',
        'ECB/EURTRY',
        'ECB/EURPHP',
        'ECB/EURZAR',
        'ECB/EURISK',
        'ECB/EURMXN',
        'ECB/EURCZK',
        'ECB/EURNOK',
        'ECB/EURAUD',
        'ECB/EURMYR',
        'ECB/EURLTL',
        'ECB/EURINR',
        'ECB/EURBGN',
        'ECB/EURCNY',
        'ECB/EURHUF',
        'ECB/EURJPY',
        'ECB/EURCAD',
        'ECB/EURRUB',
        'ECB/EURBRL',
        'ECB/EURSGD',
        'ECB/EURHRK',
        'ECB/EURILS',
        'ECB/EURNZD',
        'ECB/EURTHB',
        'ECB/EURUSD',
        'ECB/EURDKK'

    ]
    for ASSET in ALL_ASSETS:
        ASSET_NAME = ASSET.replace('/', '_')
        print(ASSET)
        try:
            print(ASSET)
            #import_data.import_data(ASSET)
            #import_data.create_train_predict(ASSET_NAME, DAY)
        except:
            print(ASSET)
