

from datetime import date, timedelta
import import_data
from list_of_assets import ALL_ASSETS
import train

if __name__ == '__main__':

    DAY = date.today() - timedelta(2)
    DAY = DAY.strftime("%Y%m%d")
    train.train_model(DAY)
    #
    # for ASSET in ALL_ASSETS:
    #     ASSET_NAME = ASSET.replace('/', '_')
    #     try:
    #         print(ASSET)
    #         #import_data.import_data(ASSET)
    #         import_data.create_train_predict(ASSET_NAME, DAY)
    #     except:
    #         print(ASSET)

    # for ASSET in ALL_ASSETS:
    #     ASSET_NAME = ASSET.replace('/', '_')
    #     train.generate_prediction(DAY, ASSET_NAME)
