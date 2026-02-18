#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
價值投注回測腳本（支援帶標籤的歷史資料）
輸入：data/high_value_bets_with_labels.csv（由 backtest_from_historical.py 生成）
輸出：回測統計報告 + 累積收益曲線圖
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ===== 設定 =====
INPUT_FILE = "data/high_value_bets_with_labels.csv"  # ← 關鍵：使用帶 is_top3 的檔案
OUTPUT_PLOT = "plots/backtest_cumulative_return.png"

def calculate_drawdown(cumulative_returns):
    """計算最大跌幅 (Max Drawdown)"""
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return max_drawdown, drawdown

def main():
    if not Path(INPUT_FILE).exists():
        print(f"❌ 找不到輸入檔案: {INPUT_FILE}")
        print("💡 請先執行: python backtest_from_historical.py")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"📊 回測資料筆數: {len(df)}")
    
    if len(df) == 0:
        print("⚠️  無歷史價值注可回測")
        return
    
    # 檢查必要欄位
    required_cols = ['is_top3', 'win_odds', 'edge', 'predicted_top3_prob']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ 缺少必要欄位: {missing}")
        print("💡 請確認檔案來自 backtest_from_historical.py")
        return
    
    # 計算命中率與基本統計
    hit_rate = df['is_top3'].mean()
    avg_edge = df['edge'].mean()
    avg_odds = df['win_odds'].mean()
    median_odds = df['win_odds'].median()
    
    # 模擬投注：每注 1 元
    df['profit'] = df.apply(
        lambda row: (row['win_odds'] - 1) if row['is_top3'] == 1 else -1,
        axis=1
    )
    total_profit = df['profit'].sum()
    total_stake = len(df)
    roi = total_profit / total_stake * 100  # ROI (%)
    
    # 累積收益（按索引排序，假設時間順序）
    df_sorted = df.sort_index().reset_index(drop=True)
    df_sorted['cumulative_profit'] = df_sorted['profit'].cumsum()
    df_sorted['cumulative_return'] = df_sorted['cumulative_profit'] / (df_sorted.index + 1) * 100
    
    # 風險指標
    max_dd, drawdown_series = calculate_drawdown(df_sorted['cumulative_profit'] + total_stake)
    sharpe_ratio = df['profit'].mean() / df['profit'].std() if df['profit'].std() != 0 else 0
    
    # 輸出詳細報告
    print("\n" + "="*50)
    print("📈 價值投注回測報告")
    print("="*50)
    print(f"總投注場次       : {total_stake:,}")
    print(f"命中率 (Top3)    : {hit_rate:.2%}")
    print(f"平均 Edge        : {avg_edge:.2%}")
    print(f"平均賠率         : {avg_odds:.2f}")
    print(f"中位數賠率       : {median_odds:.2f}")
    print("-"*50)
    print(f"總利潤           : {total_profit:,.2f} 元")
    print(f"投資報酬率 (ROI) : {roi:.2f}%")
    print(f"夏普比率         : {sharpe_ratio:.2f}")
    print(f"最大跌幅 (MDD)   : {max_dd:.2%}")
    print("="*50)
    print(f"期望值是否為正?  : {'✅ 是' if roi > 0 else '❌ 否'}")
    
    # 繪製累積收益曲線
    plt.figure(figsize=(12, 7))
    
    # 主圖：累積利潤
    plt.subplot(2, 1, 1)
    plt.plot(df_sorted.index, df_sorted['cumulative_profit'], 
             color='green', linewidth=2, label='Cumulative Profit')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('累積投注收益曲線（每注 1 元）', fontsize=14)
    plt.ylabel('累積利潤（元）')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # 下圖：最大跌幅
    plt.subplot(2, 1, 2)
    plt.fill_between(df_sorted.index, drawdown_series * 100, 0, 
                     color='red', alpha=0.3, label='Drawdown')
    plt.title('資金曲線最大跌幅 (Drawdown)', fontsize=14)
    plt.xlabel('投注序號')
    plt.ylabel('跌幅 (%)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # 儲存圖表
    Path("plots").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📉 累積收益與跌幅圖已儲存至: {OUTPUT_PLOT}")
    
    # 儲存詳細結果（供進一步分析）
    output_detail = "data/backtest_results_detailed.csv"
    df_sorted.to_csv(output_detail, index=False, encoding='utf-8')
    print(f"💾 詳細回測結果已儲存至: {output_detail}")
    
    # 高賠率 vs 低賠率分析（可選洞察）
    high_odds = df[df['win_odds'] > 10]
    low_odds = df[df['win_odds'] <= 10]
    if len(high_odds) > 0 and len(low_odds) > 0:
        print("\n🔍 賠率分組分析:")
        print(f"  高賠率 (>10): 命中率={high_odds['is_top3'].mean():.2%}, ROI={(high_odds['profit'].sum()/len(high_odds)*100):.2f}%")
        print(f"  低賠率 (≤10): 命中率={low_odds['is_top3'].mean():.2%}, ROI={(low_odds['profit'].sum()/len(low_odds)*100):.2f}%")

if __name__ == "__main__":
    main()