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

    if task_option == "前処理":
        st.sidebar.title("前処理用のデータファイルのアップロード")
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

    elif task_option == "機械学習":
        st.sidebar.title("機械学習用のデータファイルのアップロード")
        train_file = st.sidebar.file_uploader("訓練データファイル (train_df.csv)", type=["csv"], key="train")
        valid_file = st.sidebar.file_uploader("検証データファイル (validation_df.csv)", type=["csv"], key="valid")

        # メイン画面で学習回数を指定
        num_iterations = st.number_input("学習回数を指定してください", min_value=1, max_value=10000, value=1000, key="num_iterations")

        if train_file and valid_file:
            train_df = load_data(train_file, "訓練データ")
            valid_df = load_data(valid_file, "検証データ")

            if st.button("モデルのトレーニングを開始"):
                if num_iterations:
                    with st.spinner("モデルのトレーニングを実行しています..."):
                        execute_training(train_df, valid_df, model_save_dir, num_iterations)
                    st.success("モデルのトレーニングが完了しました。")
                else:
                    st.error("学習回数を指定してください。")

    elif task_option == "予測":
        st.sidebar.title("予測用のデータファイルのアップロード")
        model_file = st.sidebar.file_uploader("モデルファイル (lgbm_model.txt)", type=["txt"], key="model")
        test_file = st.sidebar.file_uploader("テストデータファイル (test_df.csv)", type=["csv"], key="test")

        if model_file and test_file:
            test_df = load_data(test_file, "テストデータ")

            if st.button("予測を実行"):
                with st.spinner("予測を実行しています..."):
                    execute_prediction(model_file, test_df, prediction_save_dir)
                st.success("予測が完了し、結果が保存されました。")

if __name__ == '__main__':
    main()
