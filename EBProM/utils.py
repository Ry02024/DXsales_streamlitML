import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import calendar
from datetime import date, timedelta, datetime
import jpholiday
# データ読み込み関数
def load_data(data_dir):
    print("データを読み込んでいます...")
    sales_history_df = pd.read_csv(data_dir + 'sales_history.csv')
    item_categories_df = pd.read_csv(data_dir + 'item_categories.csv')
    category_names_df = pd.read_csv(data_dir + 'category_names.csv')
    test_df = pd.read_csv(data_dir + 'test.csv', index_col=0)
    print("データの読み込みが完了しました。")
    return sales_history_df, item_categories_df, category_names_df, test_df

# データ前処理関数
def preprocess_data(sales_history_df, item_categories_df, category_names_df):
    print("データの前処理を開始します...")
    join_data_df = pd.merge(sales_history_df, item_categories_df, on='商品ID', how='left')
    join_data_df = pd.merge(join_data_df, category_names_df, on='商品カテゴリID', how='left')
    join_data_df = join_data_df.drop_duplicates()

    # カラム名変更
    join_data_df = join_data_df.rename(columns={'日付': 'date', '店舗ID': 'store_id', '商品ID': 'product_id',
                                                '商品価格': 'product_price', '売上個数': 'product_num',
                                                '商品カテゴリID': 'category_id', '商品カテゴリ名': 'category_name'})

    # dateをdatetime型に変換し、year, month, month_numを作成
    join_data_df['date'] = pd.to_datetime(join_data_df['date'])
    join_data_df['year'] = join_data_df['date'].dt.year
    join_data_df['month'] = join_data_df['date'].dt.month
    join_data_df['month_num'] = join_data_df['month']
    join_data_df.loc[join_data_df['year'] == 2019, 'month_num'] += 12
    join_data_df = join_data_df.drop(['date', 'year', 'month'], axis=1)

    print("データの前処理が完了しました。")
    return join_data_df

# 特徴量生成関数
def generate_features(join_data_df):
    print("特徴量を生成しています...")
    join_data_df.drop('category_name', axis=1, inplace=True, errors='ignore')
    join_data_df4_1 = join_data_df.groupby(['product_id', 'store_id', 'category_id', 'month_num']).mean()
    join_data_df4_2 = join_data_df.groupby(['product_id', 'store_id', 'category_id', 'month_num']).sum()
    join_data_df4_2['product_price'] = join_data_df4_1['product_price']

    join_data_df5 = join_data_df4_2.unstack(level=3)
    join_data_df5.to_csv('./join_data_df5_main.csv')
    join_data_df6 = pd.read_csv('./join_data_df5_main.csv', header=[0, 1, 2])

    columns_1 = ['product_id', 'store_id', 'category_id']
    columns_2 = ['product_price_' + str(i) for i in range(1, 23)]
    columns_3 = ['product_num_' + str(i) for i in range(1, 23)]
    join_data_df6.columns = columns_1 + columns_2 + columns_3

    print("特徴量の生成が完了しました。")
    return join_data_df6

def complete_catalog(join_data_df):
    product_id_pd = join_data_df[['product_id', 'category_id']].drop_duplicates().sort_values('product_id').copy()
    join_data_df7 = pd.DataFrame()

    for store_id in tqdm(sorted(join_data_df['store_id'].unique())):
        store_data = join_data_df.loc[join_data_df['store_id'] == store_id]
        merged_data = pd.merge(store_data, product_id_pd, on='product_id', how='right')
        merged_data['category_id_x'] = merged_data['category_id_y']
        merged_data.rename(columns={'category_id_x':'category_id'}, inplace=True)
        merged_data.drop(['category_id_y'], axis=1, inplace=True, errors='ignore')
        merged_data['store_id'] = store_id
        join_data_df7 = pd.concat([join_data_df7, merged_data])
    return join_data_df7

def fill_missing_values(join_data_df, test_df): #適切に平均値で保管できていない可能性
    print("欠損値を補完しています...")

    # 'product_id'ごとに、product_price_1 から product_price_22 の欠損値をその商品の平均値で補完
    price_columns = [f'product_price_{i}' for i in range(1, 23)]
    join_data_df[price_columns] = join_data_df.groupby('product_id')[price_columns].transform(lambda x: x.fillna(x.mean()))

    # product_price_1 から product_num_22 までの欠損値を0で補完
    price_and_num_columns = [f'product_price_{i}' for i in range(1, 23)] + [f'product_num_{i}' for i in range(1, 23)]
    join_data_df[price_and_num_columns] = join_data_df[price_and_num_columns].fillna(0)

    # 各列の負の値を0にする
    join_data_df[price_and_num_columns] = join_data_df[price_and_num_columns].clip(lower=0)

    # main_flag列をすべて0で初期化
    join_data_df['main_flag'] = 0

    # test_df['商品ID']に含まれるproduct_idに対してmain_flagを1に更新
    join_data_df.loc[join_data_df['product_id'].isin(test_df['商品ID']), 'main_flag'] = 1

    print("欠損値の補完が完了しました。")
    return join_data_df

# 特徴量作成関数
def fill_features(join_data_df10):
    print("追加の特徴量を生成しています...")
    # 特徴量生成1: 商品、カテゴリ、店舗ごとの平均値を生成
    target_columns = ['product_id', 'store_id', 'category_id'] + [f'product_num_{i}' for i in range(1, 23)] + [f'product_price_{i}' for i in range(1, 23)]
    join_data_df10_feats = join_data_df10[target_columns]

    # 1. 商品ごとの平均個数と平均価格
    product_ave_num = join_data_df10_feats.groupby('product_id').mean().loc[:, 'product_num_1': 'product_num_22'].mean(axis=1)
    product_ave_price = join_data_df10_feats.groupby('product_id').mean().loc[:, 'product_price_1': 'product_price_22'].mean(axis=1)

    join_data_df10.insert(3, 'product_ave_num', 0)
    join_data_df10.insert(4, 'product_ave_price', 0)
    for product_id in tqdm(product_ave_num.index):
        join_data_df10.loc[join_data_df10['product_id'] == product_id, 'product_ave_num'] = product_ave_num[product_id]
        join_data_df10.loc[join_data_df10['product_id'] == product_id, 'product_ave_price'] = product_ave_price[product_id]

    # 2. カテゴリごとの平均個数と平均価格
    category_ave_num = join_data_df10.groupby('category_id').mean().loc[:, 'product_num_1': 'product_num_22'].mean(axis=1)
    category_ave_price = join_data_df10.groupby('category_id').mean().loc[:, 'product_price_1': 'product_price_22'].mean(axis=1)

    join_data_df10.insert(5, 'category_ave_num', 0)
    join_data_df10.insert(6, 'category_ave_price', 0)
    for category_id in tqdm(category_ave_num.index):
        join_data_df10.loc[join_data_df10['category_id'] == category_id, 'category_ave_num'] = category_ave_num[category_id]
        join_data_df10.loc[join_data_df10['category_id'] == category_id, 'category_ave_price'] = category_ave_price[category_id]

    # 3. 店舗ごとの平均個数と平均価格
    store_ave_num = join_data_df10.groupby('store_id').mean().loc[:, 'product_num_1': 'product_num_22'].mean(axis=1)
    store_ave_price = join_data_df10.groupby('store_id').mean().loc[:, 'product_price_1': 'product_price_22'].mean(axis=1)

    join_data_df10.insert(7, 'store_ave_num', 0)
    join_data_df10.insert(8, 'store_ave_price', 0)
    for store_id in tqdm(store_ave_num.index):
        join_data_df10.loc[join_data_df10['store_id'] == store_id, 'store_ave_num'] = store_ave_num[store_id]
        join_data_df10.loc[join_data_df10['store_id'] == store_id, 'store_ave_price'] = store_ave_price[store_id]

    # 特徴量生成2: 商品、カテゴリ、店舗ごとの組み合わせを生成
    join_data_df10.insert(9, 'p_c_nun', join_data_df10['product_ave_num'] * join_data_df10['category_ave_num'])
    join_data_df10.insert(10, 'p_s_nun', join_data_df10['product_ave_num'] * join_data_df10['store_ave_num'])
    join_data_df10.insert(11, 'c_s_nun', join_data_df10['category_ave_num'] * join_data_df10['store_ave_num'])
    join_data_df10.insert(12, 'p_c_price', join_data_df10['product_ave_price'] * join_data_df10['category_ave_price'])
    join_data_df10.insert(13, 'p_s_price', join_data_df10['product_ave_price'] * join_data_df10['store_ave_price'])
    join_data_df10.insert(14, 'c_s_price', join_data_df10['category_ave_price'] * join_data_df10['store_ave_price'])

    # さらに複合特徴量を生成
    join_data_df10.insert(15, 'p_c_s_nun', join_data_df10['product_ave_num'] * join_data_df10['category_ave_num'] * join_data_df10['store_ave_num'])
    join_data_df10.insert(16, 'p_c_s_price', join_data_df10['product_ave_price'] * join_data_df10['category_ave_price'] * join_data_df10['store_ave_price'])

    print("追加の特徴量生成が完了しました。")
    return join_data_df10

import pandas as pd
from tqdm import tqdm

def generate_sliding_window_datasets(df,
                                     columns_to_front=None,
                                     window_size=12,
                                     n_steps=11):
    """
    スライディングウィンドウを用いて訓練データフレームとテストデータフレームを生成する関数。

    Parameters:
    - df (pd.DataFrame): 元のデータフレーム。
    - columns_to_front (list, optional): 先頭に配置するカラムのリスト。デフォルトは以下のリスト。
    - window_size (int, optional): ウィンドウのサイズ（月数）。デフォルトは12。
    - n_steps (int, optional): ウィンドウを適用するステップ数。デフォルトは11。

    Returns:
    - train_df (pd.DataFrame): スライディングウィンドウを適用した訓練データフレーム。
    - test_df (pd.DataFrame): スライディングウィンドウを適用したテストデータフレーム。
    """

    # デフォルトの columns_to_front を設定
    if columns_to_front is None:
        columns_to_front = [
            'main_flag', 'product_id', 'store_id', 'category_id',
            'product_ave_num', 'product_ave_price',
            'category_ave_num', 'category_ave_price',
            'store_ave_num', 'store_ave_price',
            'p_c_nun', 'p_s_nun', 'c_s_nun',
            'p_c_price', 'p_s_price', 'c_s_price',
            'p_c_s_nun', 'p_c_s_price'
        ]

    # 2. その他のカラムを取得（移動させたいカラムを除く）
    remaining_columns = [col for col in df.columns if col not in columns_to_front]

    # 3. 新しいカラム順序を作成
    new_column_order = columns_to_front + remaining_columns

    # データフレームを新しいカラム順に並べ替え
    df_reordered = df[new_column_order].copy()

    # テストデータの作成
    # main_flagが1の行を抽出
    tmp0_df = df_reordered.loc[df_reordered['main_flag'] == 1]

    # test_dfの作成
    tmp1_df = tmp0_df.loc[:, 'main_flag':'p_c_s_price'].copy()
    tmp2_df = tmp0_df.loc[:, 'product_num_13':'product_num_22'].copy()
    tmp2_df = tmp2_df.rename(columns=lambda x: x[:12] + str(int(x[12:])-12))
    tmp3_df = tmp0_df.loc[:, 'product_price_13':'product_price_22'].copy()
    tmp3_df = tmp3_df.rename(columns=lambda x: x[:14] + str(int(x[14:])-12))
    tmp4_df = pd.concat([tmp1_df, tmp2_df, tmp3_df], axis=1)

    # month_target を追加
    tmp4_df.insert(1, 'month_target', 12)

    test_df = tmp4_df.copy()

    # 訓練データの作成
    # main_flagが1以外の全行を対象とする
    catarog_copy2 = df_reordered.copy()

    # 予測対象の月を示す変数　month_targetを計算する関数
    def calc_month_target(n):
        tmp = n % 12
        return 12 if tmp == 0 else tmp

    # 空のデータフレームを用意
    train_df = pd.DataFrame()

    # スライディングウィンドウの適用
    for i in tqdm(range(n_steps), desc="スライディングウィンドウ適用中"):
        # 基本カラムを抽出
        tmp1_df = catarog_copy2.loc[:, 'main_flag':'p_c_s_price'].copy()

        # product_numのシフト
        product_num_cols = [f'product_num_{j}' for j in range(i+1, i+window_size+1)]
        tmp_num = catarog_copy2[product_num_cols].copy()
        tmp_num.columns = [f'product_num_{j-i}' for j in range(i+1, i+window_size+1)]

        # product_priceのシフト
        product_price_cols = [f'product_price_{j}' for j in range(i+1, i+window_size+1)]
        tmp_price = catarog_copy2[product_price_cols].copy()
        tmp_price.columns = [f'product_price_{j-i}' for j in range(i+1, i+window_size+1)]

        # シフトしたデータを結合
        tmp_combined = pd.concat([tmp1_df, tmp_num, tmp_price], axis=1)

        # 'month_target' を追加
        tmp_combined.insert(1, 'month_target', calc_month_target(i))

        # 訓練データに追加
        train_df = pd.concat([train_df, tmp_combined], ignore_index=True)

    return train_df, test_df


import pandas as pd

def generate_trend_features(train_df, test_df):
    """
    訓練データとテストデータに対して、定義された特徴量操作を適用して新しい特徴量を生成する関数。

    Parameters:
    - train_df (pd.DataFrame): 訓練データフレーム。
    - test_df (pd.DataFrame): テストデータフレーム。

    Returns:
    - train_df_gen (pd.DataFrame): 新しい特徴量が追加された訓練データフレーム。
    - test_df_gen (pd.DataFrame): 新しい特徴量が追加されたテストデータフレーム。
    """

    # 特徴量名と対応する計算方法を定義
    feature_operations = {
        'ave_num': lambda df: df.loc[:, 'product_num_1':'product_num_10'].mean(axis=1),
        'ave_price': lambda df: df.loc[:, 'product_price_1':'product_price_10'].mean(axis=1),
        'diff_10_9_num': lambda df: df['product_num_10'] - df['product_num_9'],
        'diff_10_9_price': lambda df: df['product_price_10'] - df['product_price_9'],
        'diff_10_1_num': lambda df: df['product_num_10'] - df['product_num_1'],
        'diff_10_1_price': lambda df: df['product_price_10'] - df['product_price_1'],
        'diff_10_ave_num': lambda df: df['product_num_10'] - df['ave_num'],
        'diff_10_ave_price': lambda df: df['product_price_10'] - df['ave_price']
    }

    # 特徴量生成関数
    def apply_feature_operations(df, feature_ops):
        for feature_name, operation in feature_ops.items():
            df[feature_name] = operation(df)
        return df

    # 訓練データに特徴量を生成
    train_df_copy = train_df.copy()
    train_df_gen = apply_feature_operations(train_df_copy, feature_operations)

    # テストデータに特徴量を生成
    test_df_copy = test_df.copy()
    test_df_gen = apply_feature_operations(test_df_copy, feature_operations)

    return train_df_gen, test_df_gen

import pandas as pd
from datetime import datetime, timedelta
import jpholiday

def add_calendar_features(train_df, test_df, start_date='2018-01-01', end_date='2019-12-31', predict_year_month=(2019, 12)):
    """
    訓練データとテストデータにカレンダー情報を追加する関数。

    Parameters:
    - train_df (pd.DataFrame): 訓練データフレーム。
    - test_df (pd.DataFrame): テストデータフレーム。
    - start_date (str, optional): カレンダー情報の開始日。デフォルトは '2018-01-01'。
    - end_date (str, optional): カレンダー情報の終了日。デフォルトは '2019-12-31'。
    - predict_year_month (tuple, optional): テストデータ用の年と月。デフォルトは (2019, 12)。

    Returns:
    - train_df_cal (pd.DataFrame): カレンダー情報が追加された訓練データフレーム。
    - test_df_cal (pd.DataFrame): カレンダー情報が追加されたテストデータフレーム。
    """

    # 1. 休日判定関数の定義
    def isHoliday(date):
        if jpholiday.is_holiday(date):
            return 1
        elif date.weekday() >= 5:
            return 1
        else:
            return 0

    # 2. 日数と休日数の集計
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end_datetime - start_datetime
    date_list = [start_datetime + timedelta(days=i) for i in range(delta.days + 1)]

    # 日付データフレームの作成
    all_date_df = pd.DataFrame({'datetime': date_list})
    all_date_df['is_holiday'] = all_date_df['datetime'].apply(isHoliday).astype(int)
    all_date_df['year'] = all_date_df['datetime'].dt.year
    all_date_df['month'] = all_date_df['datetime'].dt.month

    # (year, month)ごとに日数と休日数を集計
    day_holiday_num_df = all_date_df.groupby(['year', 'month']).agg(
        day_each_month=('datetime', 'count'),
        holiday_each_month=('is_holiday', 'sum')
    ).reset_index()

    # day_holiday_monthを計算
    day_holiday_num_df['day_holiday_month'] = day_holiday_num_df['day_each_month'] * day_holiday_num_df['holiday_each_month']
    day_holiday_num_df.set_index(['year', 'month'], inplace=True)

    # 2018年12月 〜　2019年10月
    day_holiday_num_df_12_10 = day_holiday_num_df.loc[(2018, 12): (2019, 10)]
    day_holiday_num_df_12_10 = day_holiday_num_df_12_10.reset_index().set_index('month')

    # 2019年12月 (予測用)
    day_holiday_num_df_12 = day_holiday_num_df.loc[(2019, 12)]
    day_holiday_num_df_12 = pd.DataFrame(day_holiday_num_df_12)

    # カレンダー情報のマッピング用 DataFrameを準備
    calendar_train_df = day_holiday_num_df_12_10.reset_index().rename(columns={'month': 'month_target'})

    # 3. 訓練データにカレンダー情報をマージ
    train_df_cal = train_df.merge(
        calendar_train_df[['month_target', 'day_each_month', 'holiday_each_month', 'day_holiday_month']],
        on='month_target',
        how='left'
    )

    # 4. テストデータにカレンダー情報を追加
    test_df_cal = test_df.merge(
        calendar_train_df[['month_target', 'day_each_month', 'holiday_each_month', 'day_holiday_month']],
        on='month_target',
        how='left'
    )

    # テストデータ用の2019年12月のカレンダー情報を追加
    test_df_cal['day_each_month'] = day_holiday_num_df_12.loc['day_each_month', (2019, 12)]
    test_df_cal['holiday_each_month'] = day_holiday_num_df_12.loc['holiday_each_month', (2019, 12)]
    test_df_cal['day_holiday_month'] = day_holiday_num_df_12.loc['day_holiday_month', (2019, 12)]

    return train_df_cal, test_df_cal

import pandas as pd

# 特徴量生成用関数
def generate_grouped_features(df, group_cols, target_col, feature_suffixes, agg_func='sum'):
    """
    グループごとの集計を計算し、新しい特徴量として追加する関数

    Parameters:
    - df (pd.DataFrame): 対象のデータフレーム
    - group_cols (list): グループ化するカラム名のリスト
    - target_col (str): 集計を計算する対象のカラム名
    - feature_suffixes (list): 生成する特徴量のサフィックス
    - agg_func (str): 使用する集計関数（'sum' または 'mean'）

    Returns:
    - pd.DataFrame: 新しい特徴量が追加されたデータフレーム
    """
    for suffix in feature_suffixes:
        feature_name = f"{agg_func}_num_{suffix}"
        product_num_col = f"{target_col}_{suffix}"
        df[feature_name] = df.groupby(group_cols)[product_num_col].transform(agg_func)
    return df

# 差分特徴量生成用関数
def generate_difference_features(df, base_feature, comparison_features, new_feature_names):
    """
    基準特徴量と比較特徴量の差分を計算し、新しい特徴量として追加する関数

    Parameters:
    - df (pd.DataFrame): 対象のデータフレーム
    - base_feature (str): 差分の基準となる特徴量名
    - comparison_features (list): 比較対象となる特徴量名のリスト
    - new_feature_names (list): 新しく生成する差分特徴量名のリスト

    Returns:
    - pd.DataFrame: 新しい差分特徴量が追加されたデータフレーム
    """
    for comp_feat, new_feat in zip(comparison_features, new_feature_names):
        df[new_feat] = df[base_feature] - df[comp_feat]
    return df

# 特徴量生成のメイン関数
def feature_engineering(train_df, test_df, group_cols, target_col, feature_suffixes, diff_features):
    """
    特徴量生成および差分計算を行うメイン関数

    Parameters:
    - train_df (pd.DataFrame): トレーニング用データフレーム
    - test_df (pd.DataFrame): テスト用データフレーム
    - group_cols (list): グループ化するカラム名のリスト
    - target_col (str): 集計を計算する対象のカラム名
    - feature_suffixes (list): 生成する特徴量のサフィックス
    - diff_features (dict): 差分特徴量の生成情報（基準特徴量と比較特徴量）

    Returns:
    - pd.DataFrame, pd.DataFrame: 特徴量が追加されたトレーニング用およびテスト用データフレーム
    """
    # 1. 特徴量生成（合計）
    train_df = generate_grouped_features(train_df, group_cols, target_col, feature_suffixes, agg_func='sum')
    test_df = generate_grouped_features(test_df, group_cols, target_col, feature_suffixes, agg_func='sum')

    # 2. 差分特徴量生成
    base_feature = 'sum_num_10'  # 基準となる特徴量
    comparison_features = ['sum_num_9', 'sum_num_8']
    new_diff_feature_names = ['ave_num_10_9', 'ave_num_10_8']

    train_df = generate_difference_features(train_df, base_feature, comparison_features, new_diff_feature_names)
    test_df = generate_difference_features(test_df, base_feature, comparison_features, new_diff_feature_names)

    return train_df, test_df

import pandas as pd
from tqdm import tqdm

def create_sales_uptrend_flag(train_df, test_df, flag_num=15, test_product_ids=[2900075]):
    """
    訓練データとテストデータに対して、売上上昇傾向フラグを作成する関数。

    Parameters:
    - train_df (pd.DataFrame): 訓練データフレーム。
    - test_df (pd.DataFrame): テストデータフレーム。
    - flag_num (int, optional): 売上増加の閾値。デフォルトは15。
    - test_product_ids (list, optional): テストデータでフラグを設定する対象の product_id のリスト。デフォルトは [2900075]。

    Returns:
    - train_df_up (pd.DataFrame): up_num_flag が追加された訓練データフレーム。
    - test_df_updated (pd.DataFrame): up_num_flag が追加されたテストデータフレーム。
    """

    # 訓練データの処理
    train_df_copy = train_df.copy()

    # diff_12_10 を計算
    train_df_copy['diff_12_10'] = train_df_copy['product_num_12'] - train_df_copy['product_num_10']

    # diff_12_10 > flag_num のインデックスを取得
    up_num_index = train_df_copy.loc[train_df_copy['diff_12_10'] > flag_num].index

    # up_num_flag を0で初期化
    train_df_copy.insert(2, 'up_num_flag', 0)

    # up_num_flag を1に設定
    train_df_copy.loc[up_num_index, 'up_num_flag'] = 1

    # diff_12_10 を削除
    train_df_up = train_df_copy.drop('diff_12_10', axis=1)

    # フラグが設定された商品の数を表示
    num_flags = train_df_up['up_num_flag'].sum()
    print(f"訓練データで up_num_flag が1に設定された商品の数: {num_flags}")

    # テストデータの処理
    test_df_updated = test_df.copy()

    # up_num_flag を0で初期化
    test_df_updated.insert(2, 'up_num_flag', 0)

    # 指定された product_id に対して up_num_flag を1に設定
    for pid in test_product_ids:
        test_df_updated.loc[test_df_updated['product_id'] == pid, 'up_num_flag'] = 1

    # 指定された product_id で product_num_10 が0以下の場合、up_num_flag を0に設定
    for pid in test_product_ids:
        condition = (test_df_updated['product_id'] == pid) & (test_df_updated['product_num_10'] <= 0)
        test_df_updated.loc[condition, 'up_num_flag'] = 0

    # 特定の product_id の一部カラムを表示（デバッグ用）
    for pid in test_product_ids:
        display_df = test_df_updated.loc[test_df_updated['product_id'] == pid, ['product_id', 'product_num_10', 'up_num_flag']]
        print(f"product_id == {pid} のデータ:")
        # print(display_df)

    return train_df_up, test_df_updated

import pandas as pd

def split_train_validation_and_sort_test(train_df_up,
                                         test_df_feats5,
                                         validation_main_flag=1,
                                         validation_month_target=12,
                                         sort_columns=['product_id', 'store_id']):
    """
    訓練データを検証用データと訓練用データに分割し、テストデータをソートする関数。

    Parameters:
    - train_df_up (pd.DataFrame): 売上上昇傾向フラグが追加された訓練データフレーム。
    - test_df_feats5 (pd.DataFrame): 前処理が完了したテストデータフレーム。
    - validation_main_flag (int, optional): 検証用データの main_flag 条件。デフォルトは 1。
    - validation_month_target (int, optional): 検証用データの month_target 条件。デフォルトは 12。
    - sort_columns (list, optional): テストデータのソートに使用するカラム。デフォルトは ['product_id', 'store_id']。

    Returns:
    - validation_df (pd.DataFrame): 検証用データフレーム。
    - train_df (pd.DataFrame): 訓練用データフレーム。
    - sorted_test_df (pd.DataFrame): ソートされたテストデータフレーム。
    """

    # 1. 訓練データの分割
    validation_df = train_df_up.loc[
        (train_df_up['main_flag'] == validation_main_flag) &
        (train_df_up['month_target'] == validation_month_target)
    ].copy()

    train_df = train_df_up.loc[
        (train_df_up['main_flag'] != validation_main_flag) |
        (train_df_up['month_target'] != validation_month_target)
    ].copy()

    # 2. テストデータのソート
    sorted_test_df = test_df_feats5.sort_values(sort_columns).copy()

    return validation_df, train_df, sorted_test_df
