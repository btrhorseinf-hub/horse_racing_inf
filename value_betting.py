# value_betting.py
"""
根據 predictions.csv 計算 Edge 和期望報酬，並輸出 value_bets_today.csv
Edge = 模型預測機率 - (1 / 市場賠率)
期望報酬 = Edge × 市場賠率
只保留 Edge > 5% 的馬匹
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_value_bets():
    # 檢查是否存在 predictions.csv
    input_path = "data/predictions.csv"
    output_path = "data/value_bets_today.csv"

    if not Path(input_path).exists():
        print(f"❌ 錯誤：{input_path} 不存在！")
        print("請先執行 POST /predict 上傳 CSV 以生成預測結果。")
        return

    try:
        df = pd.read_csv(input_path)
        print(f"✅ 成功讀取 {len(df)} 匹馬的預測結果")

        # 確保必要欄位存在
        required_cols = ['horse_name', 'win_odds', 'predicted_top3_prob']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 缺少必要欄位：{missing_cols}")
            return

        # 強制轉換數值欄位
        df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
        df['predicted_top3_prob'] = pd.to_numeric(df['predicted_top3_prob'], errors='coerce')

        # 移除無效資料
        df = df.dropna(subset=['win_odds', 'predicted_top3_prob'])
        if df.empty:
            print("❌ 沒有有效資料可處理")
            return

        # 計算 Edge 和期望報酬
        df['edge'] = df['predicted_top3_prob'] - (1 / df['win_odds'])
        df['expected_return'] = df['edge'] * df['win_odds']

        # 篩選 Edge > 0.05 (即 5%)
        df_filtered = df[df['edge'] > 0.05].copy()

        # 格式化數值（保留小數）
        df_filtered['predicted_top3_prob'] = df_filtered['predicted_top3_prob'].round(4)
        df_filtered['edge'] = df_filtered['edge'].round(4)
        df_filtered['expected_return'] = df_filtered['expected_return'].round(4)

        # 保存結果
        df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 成功生成 {len(df_filtered)} 筆高價值推薦，已保存至 {output_path}")

    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")

if __name__ == "__main__":
    calculate_value_bets()