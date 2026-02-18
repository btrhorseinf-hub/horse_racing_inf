#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
賽馬資料特徵工程腳本
輸入：historical_races.csv（需包含 race_date, horse_name, is_top3）
輸出：historical_races_with_features.csv（新增歷史表現特徵）
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = "data/historical_races.csv"
OUTPUT_PATH = "data/historical_races_with_features.csv"

def add_historical_features(df):
    """
    為每匹馬添加基於其過往賽績的特徵
    """
    # 確保日期格式正確並排序
    df = df.copy()
    df['race_date'] = pd.to_datetime(df['race_date'])
    df = df.sort_values(['horse_name', 'race_date']).reset_index(drop=True)
    
    # 初始化新特徵欄位
    new_cols = [
        'last_is_top3',
        'top3_rate_last_1',
        'top3_rate_last_3',
        'top3_rate_last_5',
        'avg_odds_last_3',
        'avg_actual_weight_last_3',
        'days_since_last_race'
    ]
    
    for col in new_cols:
        df[col] = np.nan

    # 按馬匹分組計算
    grouped = df.groupby('horse_name')
    
    for name, group in grouped:
        idxs = group.index.tolist()
        
        for i, idx in enumerate(idxs):
            if i == 0:
                # 第一場比賽，無歷史資料
                df.loc[idx, 'days_since_last_race'] = np.nan
                continue
            
            # 距離上一場天數
            last_date = df.loc[idxs[i-1], 'race_date']
            current_date = df.loc[idx, 'race_date']
            df.loc[idx, 'days_since_last_race'] = (current_date - last_date).days
            
            # 近1場
            if i >= 1:
                past_1 = group.iloc[:i]['is_top3']
                df.loc[idx, 'last_is_top3'] = past_1.iloc[-1]
                df.loc[idx, 'top3_rate_last_1'] = past_1.mean()
            
            # 近3場
            if i >= 1:
                past_3 = group.iloc[max(0, i-3):i]['is_top3']
                odds_3 = group.iloc[max(0, i-3):i]['win_odds']
                weight_3 = group.iloc[max(0, i-3):i]['actual_weight']
                df.loc[idx, 'top3_rate_last_3'] = past_3.mean()
                df.loc[idx, 'avg_odds_last_3'] = odds_3.mean()
                df.loc[idx, 'avg_actual_weight_last_3'] = weight_3.mean()
            
            # 近5場
            if i >= 1:
                past_5 = group.iloc[max(0, i-5):i]['is_top3']
                df.loc[idx, 'top3_rate_last_5'] = past_5.mean()
    
    return df

def main():
    # 檢查輸入檔案
    if not Path(INPUT_PATH).exists():
        print(f"❌ 輸入檔案不存在: {INPUT_PATH}")
        return
    
    # 讀取資料
    print(f"讀取歷史資料: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    
    # 驗證必要欄位
    required_cols = {'race_date', 'horse_name', 'is_top3', 'win_odds', 'actual_weight'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"❌ 缺少必要欄位: {missing}")
        return
    
    # 添加特徵
    print("正在計算歷史特徵...")
    df_enhanced = add_historical_features(df)
    
    # 儲存結果
    df_enhanced.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"✅ 特徵工程完成！輸出至: {OUTPUT_PATH}")
    print(f"📊 新增特徵: {df_enhanced.columns[-8:].tolist()}")

if __name__ == "__main__":
    main()