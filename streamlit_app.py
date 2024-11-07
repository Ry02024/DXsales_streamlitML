import sys
import os
import streamlit as st
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from EBProM.execute import execute_preprocessing, execute_training, execute_prediction

# 各タスクの完了フラグを初期化
if "preprocessing_done" not in st.session_state:
    st.session_state["preprocessing_done"] = False
if "training_done" not in st.session_state:
    st.session_state["training_done"] = False
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False

# データをロードするヘルパー関数
def load_data(uploaded_file, description):
    with st.spinner(f"{description}を読み込んでいます..."):
        return pd.read_csv(uploaded_file)

# 前処理関数
def execute_preprocessing(sales_df, item_df, category_df, test_df, save_dir):
    # 前処理処理の例
    train_df = pd.DataFrame()  # 実際の処理内容を入れてください
    validation_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # ファイルの保存
    train_path = os.path.join(save_dir, 'train_df.csv')
    validation_path = os.path.join(save_dir, 'validation_df.csv')
    test_path = os.path.join(save_dir, 'test_df.csv')
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 前処理完了フラグを設定
    st.session_state["preprocessing_done"] = True
    st.session_state["train_path"] = train_path
    st.session_state["validation_path"] = validation_path
    st.session_state["test_path"] = test_path

# 学習関数
def execute_training(train_df, valid_df, model_save_dir, num_iterations=1000):
    # 学習処理の例
    model_path = os.path.join(model_save_dir, 'lgbm_model.txt')
    # モデルの保存
    with open(model_path, 'w') as f:
        f.write("dummy model data")  # 実際のモデル保存処理に置き換えてください

    # 学習完了フラグを設定
    st.session_state["training_done"] = True
    st.session_state["model_path"] = model_path

# 予測関数
def execute_prediction(model_file, test_df, prediction_save_dir):
    # 予測処理の例
    prediction_path = os.path.join(prediction_save_dir, 'predictions.csv')
    pd.DataFrame({"predictions": [0, 1, 0, 1]}).to_csv(prediction_path, index=False)  # 実際の予測処理に置き換えてください

    # 予測完了フラグを設定
    st.session_state["prediction_done"] = True
    st.session_state["prediction_path"] = prediction_path

# Streamlitアプリケーション
def main():
    st.title('モデルのトレーニングと予測')

    # 各ディレクトリの指定
    save_dir = st.text_input("前処理データ保存ディレクトリ", value="Data/")
    model_save_dir = st.text_input("モデル保存ディレクトリ", value="Models/")
    prediction_save_dir = st.text_input("予測結果保存ディレクトリ", value="Predictions/")

    # タスク選択
    task_option = st.sidebar.radio("実行するタスクを選択してください", ("前処理", "機械学習", "予測"))

    if task_option == "前処理":
        sales_file = st.sidebar.file_uploader("売上データファイル (sales_history.csv)", type=["csv"], key="sales")
        item_file = st.sidebar.file_uploader("商品データファイル (item_categories.csv)", type=["csv"], key="item")
        category_file = st.sidebar.file_uploader("カテゴリデータファイル (category_names.csv)", type=["csv"], key="category")
        test_file = st.sidebar.file_uploader("テストデータファイル (test.csv)", type=["csv"], key="test")

        if sales_file and item_file and category_file and test_file:
            sales_df = load_data(sales_file, "売上データ")
            item_df = load_data(item_file, "商品データ")
            category_df = load_data(category_file, "カテゴリデータ")
            test_df = load_data(test_file, "テストデータ")

            if st.button("前処理と特徴量生成を実行"):
                with st.spinner("前処理と特徴量生成を実行しています..."):
                    execute_preprocessing(sales_df, item_df, category_df, test_df, save_dir)
                st.success("前処理と特徴量生成が完了しました。")

        # 前処理が完了した場合、ダウンロードボタンを表示
        if st.session_state["preprocessing_done"]:
            with open(st.session_state["train_path"], "rb") as file:
                st.download_button(label="Download train_df.csv", data=file, file_name="train_df.csv", mime="text/csv")
            with open(st.session_state["validation_path"], "rb") as file:
                st.download_button(label="Download validation_df.csv", data=file, file_name="validation_df.csv", mime="text/csv")
            with open(st.session_state["test_path"], "rb") as file:
                st.download_button(label="Download test_df.csv", data=file, file_name="test_df.csv", mime="text/csv")

    elif task_option == "機械学習":
        train_file = st.sidebar.file_uploader("訓練データファイル (train_df.csv)", type=["csv"], key="train")
        valid_file = st.sidebar.file_uploader("検証データファイル (validation_df.csv)", type=["csv"], key="valid")
        num_iterations = st.number_input("学習回数を指定", min_value=1, value=1000)

        if train_file and valid_file:
            train_df = load_data(train_file, "訓練データ")
            valid_df = load_data(valid_file, "検証データ")

            if st.button("モデルのトレーニングを開始"):
                with st.spinner("モデルのトレーニングを実行しています..."):
                    execute_training(train_df, valid_df, model_save_dir, num_iterations)
                st.success("モデルのトレーニングが完了しました。")

        # 学習が完了した場合、ダウンロードボタンを表示
        if st.session_state["training_done"]:
            with open(st.session_state["model_path"], "rb") as file:
                st.download_button(label="Download lgbm_model.txt", data=file, file_name="lgbm_model.txt", mime="text/plain")

    elif task_option == "予測":
        model_file = st.sidebar.file_uploader("モデルファイル (lgbm_model.txt)", type=["txt"], key="model")
        test_file = st.sidebar.file_uploader("テストデータファイル (test_df.csv)", type=["csv"], key="test")

        if model_file and test_file:
            test_df = load_data(test_file, "テストデータ")

            if st.button("予測を実行"):
                with st.spinner("予測を実行しています..."):
                    execute_prediction(model_file, test_df, prediction_save_dir)
                st.success("予測が完了しました。")

        # 予測が完了した場合、ダウンロードボタンを表示
        if st.session_state["prediction_done"]:
            with open(st.session_state["prediction_path"], "rb") as file:
                st.download_button(label="Download predictions.csv", data=file, file_name="predictions.csv", mime="text/csv")

if __name__ == '__main__':
    main()
