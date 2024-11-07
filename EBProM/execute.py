import os
import pandas as pd
from lightgbm import Booster
import tempfile
import streamlit as st
from .utils import *  # 相対インポートで utils をインポート
from .machine_learning import *

# 前処理と特徴量生成を実行するメイン処理関数
def execute_preprocessing(sales_df, item_df, category_df, test_df, save_dir):
    try:
        st.write("データの結合と前処理を実行中...")
        joined_data = preprocess_data(sales_df, item_df, category_df)
        st.success("データの結合と前処理が完了しました。")

        st.write("特徴量生成を実行中...")
        joined_data = generate_features(joined_data)
        st.success("特徴量生成が完了しました。")

        st.write("カタログデータの生成中...")
        catalog_df = complete_catalog(joined_data)
        st.success("カタログデータの生成が完了しました。")

        st.write("欠損値を補完中...")
        catalog_df_filled = fill_missing_values(catalog_df, test_df)
        st.success("欠損値補完が完了しました。")

        st.write("追加の特徴量補完を実行中...")
        catalog_filled_feats = fill_features(catalog_df_filled)
        st.success("追加の特徴量補完が完了しました。")

        st.write("スライディングウィンドウを使用してデータセットを生成中...")
        train_df, test_df = generate_sliding_window_datasets(catalog_filled_feats)
        st.success("スライディングウィンドウを使用したデータセット生成が完了しました。")

        st.write("トレンド特徴量を生成中...")
        train_df, test_df = generate_trend_features(train_df, test_df)
        st.success("トレンド特徴量の生成が完了しました。")

        st.write("カレンダー情報を追加中...")
        train_df, test_df = add_calendar_features(train_df, test_df, start_date='2018-01-01', end_date='2019-12-31', predict_year_month=(2019, 12))
        st.success("カレンダー情報の追加が完了しました。")

        st.write("データの分割とソートを実行中...")
        validation_df, train_df, test_df = split_train_validation_and_sort_test(train_df, test_df, validation_main_flag=1, validation_month_target=12)
        st.success("データの分割とソートが完了しました。")

        # データの保存
        st.write("前処理が完了し、データを保存中...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_df.to_csv(os.path.join(save_dir, 'train_df.csv'), index=False)
        validation_df.to_csv(os.path.join(save_dir, 'validation_df.csv'), index=False)
        test_df.to_csv(os.path.join(save_dir, 'test_df.csv'), index=False)
        st.success(f"前処理が完了し、データが {save_dir} に保存されました。")

    except KeyError as e:
        st.error(f"KeyErrorが発生しました: {e}")
        st.error("データの列名を確認してください。")

# モデルのトレーニングを実行する関数
def execute_training(train_df, valid_df, model_save_dir, num_iterations):
    with st.spinner("データセットをセットアップしています..."):
        lgb_train, lgb_eval = set_data_set(train_df, valid_df)
        st.success("データセットのセットアップが完了しました。")

    with st.spinner("LightGBMでモデルをトレーニング中..."):
        gbm = train_by_lightgbm(lgb_train, lgb_eval, num_iterations)
        st.success("モデルのトレーニングが完了しました。")

    # モデルの保存
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_path = os.path.join(model_save_dir, 'lgbm_model.txt')
    gbm.save_model(model_path)
    st.session_state["model_path"] = model_path  # セッションステートに保存
    st.success(f"モデルが {model_save_dir} に 'lgbm_model.txt' として保存されました。")

# 推論を実行する関数
def execute_prediction(model_file, test_df, prediction_save_dir):
    st.write("モデルをロードしています...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(model_file.read())
        model_path = tmp_file.name

    gbm = Booster(model_file=model_path)
    st.success("モデルのロードが完了しました。")

    st.write("予測を実行中...")
    predictions = gbm.predict(test_df)
    st.success("予測が完了しました。")

    # 予測結果の保存
    if not os.path.exists(prediction_save_dir):
        os.makedirs(prediction_save_dir)
    prediction_path = os.path.join(prediction_save_dir, 'predictions.csv')
    pd.DataFrame(predictions, columns=["predictions"]).to_csv(prediction_path, index=False)
    st.success(f"予測結果が {prediction_save_dir} に 'predictions.csv' として保存されました。")
