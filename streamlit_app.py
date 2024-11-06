import sys
import os
import streamlit as st
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from EBProM.execute import execute_preprocessing, execute_training, execute_prediction

# ファイルアップロード後にメッセージを表示しながら処理
def load_data(uploaded_file, description):
    with st.spinner(f"{description}を読み込んでいます..."):
        return pd.read_csv(uploaded_file)

def main():
    st.title('モデルのトレーニングと予測')

    # 各ディレクトリの指定
    st.write("データの保存先ディレクトリを指定してください（例：/content/drive/MyDrive/保存先）")
    save_dir = st.text_input("前処理データ保存ディレクトリ", value="Data/")
    model_save_dir = st.text_input("モデル保存ディレクトリ", value="Models/")
    prediction_save_dir = st.text_input("予測結果保存ディレクトリ", value="Predictions/")

    # 前処理、機械学習、予測の選択
    task_option = st.sidebar.radio("実行するタスクを選択してください", ("前処理", "機械学習", "予測"))

    if task_option == "前処理":
        st.sidebar.title("前処理用のデータファイルのアップロード")
        sales_file = st.sidebar.file_uploader("売上データファイル (sales_history.csv)", type=["csv"], key="sales")
        item_file = st.sidebar.file_uploader("商品データファイル (item_categories.csv)", type=["csv"], key="item")
        category_file = st.sidebar.file_uploader("カテゴリデータファイル (category_names.csv)", type=["csv"], key="category")
        test_file = st.sidebar.file_uploader("テストデータファイル (test.csv)", type=["csv"], key="test")

        if sales_file and item_file and category_file and test_file:
            sales_df = pd.read_csv(sales_file)
            item_df = pd.read_csv(item_file)
            category_df = pd.read_csv(category_file)
            test_df = pd.read_csv(test_file)

            if st.button("前処理と特徴量生成を実行"):
                with st.spinner("前処理と特徴量生成を実行しています..."):
                    execute_preprocessing(sales_df, item_df, category_df, test_df, save_dir)
                st.success("前処理と特徴量生成が完了しました。")

    elif task_option == "機械学習":
        st.sidebar.title("機械学習用のデータファイルのアップロード")
        train_file = st.sidebar.file_uploader("訓練データファイル (train_df.csv)", type=["csv"], key="train")
        valid_file = st.sidebar.file_uploader("検証データファイル (validation_df.csv)", type=["csv"], key="valid")

        if train_file and valid_file:
            train_df = pd.read_csv(train_file)
            valid_df = pd.read_csv(valid_file)

            if st.button("モデルのトレーニングを開始"):
                with st.spinner("モデルのトレーニングを実行しています..."):
                    execute_training(train_df, valid_df, model_save_dir)
                st.success("モデルのトレーニングが完了しました。")

    elif task_option == "予測":
        st.sidebar.title("予測用のデータファイルのアップロード")
        model_file = st.sidebar.file_uploader("モデルファイル (lgbm_model.txt)", type=["txt"], key="model")
        test_file = st.sidebar.file_uploader("テストデータファイル (test_df.csv)", type=["csv"], key="test")

        if model_file and test_file:
            test_df = pd.read_csv(test_file)

            if st.button("予測を実行"):
                with st.spinner("予測を実行しています..."):
                    execute_prediction(model_file, test_df, prediction_save_dir)
                st.success("予測が完了し、結果が保存されました。")

if __name__ == '__main__':
    main()
