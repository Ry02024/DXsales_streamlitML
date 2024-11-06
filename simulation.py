import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sim_visualize(join_data_df, test_df, sub):
    # main_flag列をすべて0で初期化
    join_data_df['main_flag'] = 0

    # test_df['商品ID']に含まれるproduct_idに対してmain_flagを1に更新
    join_data_df.loc[join_data_df['product_id'].isin(test_df['product_id']), 'main_flag'] = 1

    # 実際の全店舗の販売個数を集計（去年の12月の実績）
    total_actual_sales = join_data_df[(join_data_df['main_flag'] == 1) & (join_data_df['month_num'] == 12)]['product_num'].sum()

    merged_df2 = pd.merge(test_df, sub, left_index=True, right_index=True, how='inner')

    comp_df = merged_df2[['store_id','category_id','product_id',1]]

    # 予測された全店舗の販売個数を集計
    total_predicted_sales = comp_df[1].sum()

    # 実際の販売個数と予測販売個数をデータフレームにまとめる
    summary_df = pd.DataFrame({
        'Sales Type': ['Actual Sales', 'Predicted Sales'],
        'Total Sales': [total_actual_sales, total_predicted_sales]
    })

    # 全体の販売個数の比較を棒グラフで可視化
    summary_df.set_index('Sales Type').plot(kind='bar', figsize=(6, 4), legend=False)
    plt.title('Total Actual vs Predicted Sales (All Stores)')
    plt.ylabel('Total Sales')
    plt.show()

    # 集計結果を表示
    print(summary_df)

    # 実際の販売個数を店舗ごとに集計
    actual_sales = join_data_df[(join_data_df['main_flag'] == 1) & (join_data_df['month_num'] == 12)].groupby('store_id')['product_num'].sum()

    # 予測された販売個数を店舗ごとに集計
    predicted_sales = comp_df.groupby('store_id')[1].sum()

    # 実際の販売個数と予測された販売個数を結合
    comparison_df = pd.DataFrame({
        'Actual Sales': actual_sales,
        'Predicted Sales': predicted_sales
    })
    # 店舗ごとの販売個数を棒グラフで可視化
    comparison_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Actual vs Predicted Sales by Store')
    plt.xlabel('Store ID')
    plt.ylabel('Sales')
    plt.show()
