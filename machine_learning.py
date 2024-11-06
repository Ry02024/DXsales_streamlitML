import pandas as pd
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

def set_data_set(train_df, validation_df):
    # xとyに分割
    train_x_df = train_df.drop(['product_num_11', 'product_num_12', 'product_price_11', 'product_price_12'], axis=1).copy()
    train_y_df = pd.DataFrame(train_df['product_num_12'])
    validation_x_df = validation_df.drop(['product_num_11', 'product_num_12', 'product_price_11', 'product_price_12'], axis=1).copy()
    validation_y_df = pd.DataFrame(validation_df['product_num_12'])

    # LightGBM用のデータセットに変換
    lgb_train = lgb.Dataset(train_x_df, train_y_df)
    lgb_eval = lgb.Dataset(validation_x_df, validation_y_df)

    return lgb_train, lgb_eval

def train_by_lightgbm(lgb_train, lgb_eval, round=1000):
    # ハイパーパラメータを設定
    params = {'metric': 'rmse',
            }

    callbacks = [
        log_evaluation(period=500),  # 100イテレーションごとに結果を表示
        early_stopping(stopping_rounds=500, verbose=True)
        ]
    # 学習
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_train,lgb_eval],
                    num_boost_round=round,
                    callbacks = callbacks,
            )
    # モデルをファイルに保存
    # gbm.save_model('lightgbm_model_22_2.txt')
    return gbm
