# backtest_from_historical.py
#!/usr/bin/env python3
"""
從歷史資料直接計算價值投注回測（無需預測）
假設：使用模型預測機率（來自 predictions_train.csv）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HIST_FILE = "data/historical_races_with_features.csv"
PRED_FILE = "data/predictions_train.csv"
OUTPUT_HIGH_VALUE = "data/high_value_bets_with_labels.csv"
EDGE_THRESHOLD = 0.05

def main():
    # 讀取資料
    hist = pd.read_csv(HIST_FILE)
    pred = pd.read_csv(PRED_FILE)
    
    # 合併預測機率到歷史資料
    df = hist.merge(
        pred[['predicted_top3_prob']],
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    print(f"總歷史筆數: {len(df)}")
    
    # 計算隱含機率與 Edge
    df['implied_prob'] = 1 / df['win_odds']
    df['edge'] = df['predicted_top3_prob'] - df['implied_prob']
    df['expected_return'] = (
        df['predicted_top3_prob'] * (df['win_odds'] - 1) -
        (1 - df['predicted_top3_prob'])
    )
    
    # 篩選高價值注
    high_value = df[df['edge'] > EDGE_THRESHOLD].copy()
    print(f"高價值注筆數: {len(high_value)}")
    
    if len(high_value) == 0:
        print("⚠️  無符合條件的高價值注")
        return
    
    # 儲存（此版本包含 is_top3！）
    high_value.to_csv(OUTPUT_HIGH_VALUE, index=False, encoding='utf-8')
    print(f"✅ 已儲存帶標籤的高價值注至: {OUTPUT_HIGH_VALUE}")
    
    # 計算 ROI
    high_value['profit'] = np.where(
        high_value['is_top3'] == 1,
        high_value['win_odds'] - 1,
        -1
    )
    total_profit = high_value['profit'].sum()
    roi = total_profit / len(high_value) * 100
    
    print("\n📈 回測結果:")
    print(f"命中率: {high_value['is_top3'].mean():.2%}")
    print(f"平均賠率: {high_value['win_odds'].mean():.2f}")
    print(f"總利潤: {total_profit:.2f}")
    print(f"ROI: {roi:.2f}%")
    
    # 繪圖
    Path("plots").mkdir(exist_ok=True)
    high_value_sorted = high_value.sort_index()
    high_value_sorted['cumulative_profit'] = high_value_sorted['profit'].cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(high_value_sorted['cumulative_profit'], color='green')
    plt.title('歷史高價值注累積收益（模擬）')
    plt.xlabel('投注序號')
    plt.ylabel('累積利潤（每注1元）')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig("plots/backtest_historical.png", dpi=150)
    plt.close()
    print("📉 收益曲線已儲存至: plots/backtest_historical.png")

if __name__ == "__main__":
    main()