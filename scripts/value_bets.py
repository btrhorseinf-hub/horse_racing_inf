# value_betting.py
import pandas as pd
import numpy as np

def calculate_edge_and_return(df):
    df['edge'] = df['predicted_top3_prob'] - (1 / df['win_odds'])
    df['expected_return'] = df['edge'] * df['win_odds']

    # 只保留 Edge > 0.05 的馬匹
    df_filtered = df[df['edge'] > 0.05].copy()

    # 格式化數值
    df_filtered['predicted_top3_prob'] = df_filtered['predicted_top3_prob'].round(4)
    df_filtered['edge'] = df_filtered['edge'].round(4)
    df_filtered['expected_return'] = df_filtered['expected_return'].round(4)

    return df_filtered

if __name__ == "__main__":
    # 讀取預測結果
    try:
        predictions = pd.read_csv("data/predictions.csv")
        print("讀取成功！")

        # 計算 Edge 和期望報酬
        result = calculate_edge_and_return(predictions)
        
        # 保存為 value_bets_today.csv
        result.to_csv("data/value_bets_today.csv", index=False)
        print("價值投注推薦已生成！")
        
    except FileNotFoundError:
        print("❌ error: data/predictions.csv 不存在，請先執行馬匹勝率預測 API")